#!/usr/bin/env python3
"""Run a context-bundle baseline: sample questions, answer from fixed evidence, then evaluate."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from evaluate_rag_answers import evaluate_file
from table_metadata_index import normalize_table_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_QUESTIONS_DIR = REPO_ROOT / "questions_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "questions_final_test" / "context_bundle_baseline"
DEFAULT_PUBMED_TABLE_MAP = REPO_ROOT / "data" / "Pharma" / "pubmed-drugbank-tables.gt"
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_EVAL_MODEL = "openai/gpt-5-chat"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_EVIDENCE_COUNT = 5
DEFAULT_MAX_EVIDENCE_CHARS = 2400


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    modality: str  # "passage" | "table"
    text: str
    source_file: str
    origin_question_id: str
    origin_question: str
    is_ground_truth: bool


@dataclass(frozen=True)
class QuestionRecord:
    source_file: str
    question_type: str
    row: dict[str, Any]
    ground_truth_items: list[EvidenceItem]


@dataclass
class OpenRouterConfig:
    client: OpenAI
    headers: dict[str, str]


@dataclass
class AnswerResult:
    idx: int
    direct_answer: str
    reasoning_answer: str
    system_answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    processing_time: float
    evidence_items: list[EvidenceItem]


_THREAD_LOCAL = threading.local()


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _parse_json_if_possible(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _to_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    parsed = _parse_json_if_possible(text)
    if isinstance(parsed, list):
        return parsed
    if parsed is not None:
        return [parsed]
    if "," in text:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return [text]


def _to_text_list(value: object) -> list[str]:
    return [_safe_text(item) for item in _to_list(value) if _safe_text(item)]


def _serialize_table_payload(item: Any) -> str:
    if isinstance(item, (dict, list)):
        return json.dumps(item, ensure_ascii=False)
    return _safe_text(item)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16].rstrip() + "\n...[truncated]"


def _normalize_table_id(raw_table_id: str, fallback: str) -> str:
    text = _safe_text(raw_table_id)
    if not text:
        text = fallback
    return normalize_table_name(text)


def _extract_row_evidence_items(
    row: dict[str, Any],
    *,
    source_file: str,
    is_ground_truth: bool,
    max_chars: int,
) -> list[EvidenceItem]:
    question_id = _safe_text(row.get("question_id"))
    question_text = _safe_text(row.get("question"))
    evidence: list[EvidenceItem] = []

    passage_texts = _to_list(row.get("pubmed_text"))
    passage_ids = _to_text_list(row.get("pubmed_id"))
    for idx, text_obj in enumerate(passage_texts):
        text = _truncate_text(_safe_text(text_obj), max_chars=max_chars)
        if not text:
            continue
        pubmed_id = passage_ids[idx] if idx < len(passage_ids) else ""
        evidence_id = pubmed_id or f"{question_id}_passage_{idx + 1}"
        evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                modality="passage",
                text=text,
                source_file=source_file,
                origin_question_id=question_id,
                origin_question=question_text,
                is_ground_truth=is_ground_truth,
            )
        )

    table_payloads = _to_list(row.get("table_content"))
    table_ids = _to_text_list(row.get("table_id"))
    for idx, payload in enumerate(table_payloads):
        text = _truncate_text(_serialize_table_payload(payload), max_chars=max_chars)
        if not text:
            continue
        table_id_raw = table_ids[idx] if idx < len(table_ids) else ""
        if not table_id_raw and isinstance(payload, dict):
            table_id_raw = _safe_text(payload.get("table_name"))
        fallback_id = f"{question_id}_table_{idx + 1}"
        table_id = _normalize_table_id(table_id_raw, fallback=fallback_id)
        evidence.append(
            EvidenceItem(
                evidence_id=table_id,
                modality="table",
                text=text,
                source_file=source_file,
                origin_question_id=question_id,
                origin_question=question_text,
                is_ground_truth=is_ground_truth,
            )
        )

    return evidence


def _question_type_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("_processed", "")


def _load_question_records(
    questions_dir: Path,
    max_evidence_chars: int,
) -> tuple[dict[str, list[QuestionRecord]], dict[str, list[EvidenceItem]], list[EvidenceItem]]:
    records_by_type: dict[str, list[QuestionRecord]] = {}
    pool_by_file: dict[str, list[EvidenceItem]] = {}
    global_pool: list[EvidenceItem] = []

    question_files = sorted(questions_dir.glob("*_processed.csv"))
    if not question_files:
        raise FileNotFoundError(f"No *_processed.csv files found in {questions_dir}")

    for csv_path in question_files:
        df = pd.read_csv(csv_path)
        source_file = csv_path.name
        q_type = _question_type_from_path(csv_path)
        records: list[QuestionRecord] = []
        pool: list[EvidenceItem] = []
        for row_dict in df.to_dict(orient="records"):
            gt_items = _extract_row_evidence_items(
                row_dict,
                source_file=source_file,
                is_ground_truth=True,
                max_chars=max_evidence_chars,
            )
            records.append(
                QuestionRecord(
                    source_file=source_file,
                    question_type=q_type,
                    row=row_dict,
                    ground_truth_items=gt_items,
                )
            )
            pool.extend(
                EvidenceItem(
                    evidence_id=item.evidence_id,
                    modality=item.modality,
                    text=item.text,
                    source_file=item.source_file,
                    origin_question_id=item.origin_question_id,
                    origin_question=item.origin_question,
                    is_ground_truth=False,
                )
                for item in gt_items
            )
        records_by_type[q_type] = records
        pool_by_file[source_file] = pool
        global_pool.extend(pool)

    return records_by_type, pool_by_file, global_pool


def _allocate_quotas(total_samples: int, capacities: dict[str, int]) -> dict[str, int]:
    types = sorted(capacities)
    if total_samples <= 0:
        return {k: 0 for k in types}

    quotas = {k: 0 for k in types}
    if total_samples >= len(types):
        for t in types:
            if capacities[t] > 0:
                quotas[t] = 1

    assigned = sum(quotas.values())
    remaining = max(0, total_samples - assigned)

    while remaining > 0:
        progress = False
        for t in types:
            if remaining == 0:
                break
            if quotas[t] >= capacities[t]:
                continue
            quotas[t] += 1
            remaining -= 1
            progress = True
        if not progress:
            break
    return quotas


def _sample_records_stratified(
    records_by_type: dict[str, list[QuestionRecord]],
    sample_size: int,
    seed: int,
) -> list[QuestionRecord]:
    capacities = {q_type: len(records) for q_type, records in records_by_type.items()}
    quotas = _allocate_quotas(total_samples=sample_size, capacities=capacities)
    rng = random.Random(seed)
    sampled: list[QuestionRecord] = []
    for q_type in sorted(records_by_type):
        records = records_by_type[q_type]
        k = min(quotas[q_type], len(records))
        if k <= 0:
            continue
        sampled.extend(rng.sample(records, k))
    rng.shuffle(sampled)
    return sampled


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _relevance_score(query: str, candidate_query: str) -> float:
    query_tokens = _tokenize(query)
    candidate_tokens = _tokenize(candidate_query)
    overlap = 0.0
    if query_tokens:
        overlap = len(query_tokens.intersection(candidate_tokens)) / float(len(query_tokens))
    ratio = SequenceMatcher(None, query.lower(), candidate_query.lower()).ratio()
    return overlap * 0.7 + ratio * 0.3


def _choose_evidence_bundle(
    record: QuestionRecord,
    *,
    evidence_count: int,
    pool_by_file: dict[str, list[EvidenceItem]],
    global_pool: list[EvidenceItem],
) -> list[EvidenceItem]:
    row = record.row
    question_id = _safe_text(row.get("question_id"))
    question_text = _safe_text(row.get("question"))
    gt_items = list(record.ground_truth_items)

    if not gt_items:
        return []

    allowed_modalities = {item.modality for item in gt_items}
    selected: list[EvidenceItem] = gt_items[:evidence_count]
    selected_ids = {item.evidence_id for item in selected}

    def _candidate_score(item: EvidenceItem) -> float:
        return _relevance_score(question_text, item.origin_question)

    def _iter_candidates(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
        filtered: list[EvidenceItem] = []
        for item in items:
            if item.modality not in allowed_modalities:
                continue
            if item.origin_question_id == question_id:
                continue
            if item.evidence_id in selected_ids:
                continue
            filtered.append(item)
        filtered.sort(key=_candidate_score, reverse=True)
        return filtered

    local_candidates = _iter_candidates(pool_by_file.get(record.source_file, []))
    for item in local_candidates:
        if len(selected) >= evidence_count:
            break
        selected.append(item)
        selected_ids.add(item.evidence_id)

    if len(selected) < evidence_count:
        global_candidates = _iter_candidates(global_pool)
        for item in global_candidates:
            if len(selected) >= evidence_count:
                break
            selected.append(item)
            selected_ids.add(item.evidence_id)

    return selected[:evidence_count]


def build_openrouter_config() -> OpenRouterConfig:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for OpenRouter.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    return OpenRouterConfig(client=client, headers=headers)


def _get_thread_openrouter_config() -> OpenRouterConfig:
    config = getattr(_THREAD_LOCAL, "openrouter_config", None)
    if config is None:
        config = build_openrouter_config()
        _THREAD_LOCAL.openrouter_config = config
    return config


def _extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        start = match.start()
        try:
            candidate, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return json.dumps(candidate, ensure_ascii=False)
    return None


def _extract_json_string_field(text: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    raw = match.group(1)
    try:
        return bytes(raw, "utf-8").decode("unicode_escape").strip()
    except Exception:
        return raw.strip()


def _answer_question_with_evidence(
    *,
    idx: int,
    question: str,
    evidence_items: list[EvidenceItem],
    model: str,
) -> AnswerResult:
    config = _get_thread_openrouter_config()

    evidence_lines: list[str] = []
    for i, item in enumerate(evidence_items, start=1):
        evidence_lines.append(
            f"[Evidence {i}] id={item.evidence_id} modality={item.modality} "
            f"target_evidence={'true' if item.is_ground_truth else 'false'}\n{item.text}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical QA assistant. Use only the provided evidence snippets. "
                "Return strict JSON with keys direct_answer and reasoning_answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Evidence bundle:\n{evidence_block}\n\n"
                "Rules:\n"
                "- target_evidence=true marks which evidence belongs to the current question. "
                "It is not an answer label.\n"
                "- If evidence is insufficient, say so explicitly.\n"
                "- direct_answer should be concise and final.\n"
                "- reasoning_answer should cite evidence IDs used."
            ),
        },
    ]

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            started = time.perf_counter()
            response = config.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                extra_headers=config.headers,
            )
            duration = time.perf_counter() - started
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError("Answer generation failed with no response.")

    content = (response.choices[0].message.content or "").strip()
    direct_answer = ""
    reasoning_answer = ""

    json_text = _extract_json_object(content)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                direct_answer = _safe_text(parsed.get("direct_answer"))
                reasoning_answer = _safe_text(parsed.get("reasoning_answer"))
        except json.JSONDecodeError:
            pass

    if not direct_answer:
        direct_answer = _extract_json_string_field(content, "direct_answer")
    if not reasoning_answer:
        reasoning_answer = _extract_json_string_field(content, "reasoning_answer")

    if not direct_answer and not reasoning_answer:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        direct_answer = lines[0] if lines else content
        reasoning_answer = content

    if not direct_answer and reasoning_answer:
        direct_answer = reasoning_answer.split("\n", 1)[0].strip()
    if not direct_answer and not reasoning_answer:
        direct_answer = "Insufficient information from provided evidence."
        reasoning_answer = direct_answer

    system_answer = direct_answer
    if reasoning_answer:
        system_answer = f"{direct_answer}\n\n{reasoning_answer}".strip()

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    return AnswerResult(
        idx=idx,
        direct_answer=direct_answer,
        reasoning_answer=reasoning_answer,
        system_answer=system_answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        processing_time=duration,
        evidence_items=evidence_items,
    )


def _build_output_row(record: QuestionRecord, answer: AnswerResult) -> dict[str, Any]:
    row = dict(record.row)
    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "modality": item.modality,
            "is_ground_truth": item.is_ground_truth,
            "origin_question_id": item.origin_question_id,
            "source_file": item.source_file,
        }
        for item in answer.evidence_items
    ]

    passages_used = [item.evidence_id for item in answer.evidence_items if item.modality == "passage"]
    tables_used = [normalize_table_name(item.evidence_id) for item in answer.evidence_items if item.modality == "table"]

    row.update(
        {
            "question_type": record.question_type,
            "source_file": record.source_file,
            "system_answer": answer.system_answer,
            "direct_system_answer": answer.direct_answer,
            "reasoning_system_answer": answer.reasoning_answer,
            "passages_used": ", ".join(passages_used),
            "tables_used": ", ".join(tables_used),
            "evidence_bundle_json": json.dumps(evidence_payload, ensure_ascii=False),
            "answer_prompt_tokens": answer.prompt_tokens,
            "answer_completion_tokens": answer.completion_tokens,
            "answer_total_tokens": answer.total_tokens,
            "answer_processing_time": answer.processing_time,
        }
    )
    return row


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    questions_dir = Path(args.questions_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_type, pool_by_file, global_pool = _load_question_records(
        questions_dir=questions_dir,
        max_evidence_chars=args.max_evidence_chars,
    )
    sampled_records = _sample_records_stratified(
        records_by_type=records_by_type,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if not sampled_records:
        raise RuntimeError("No rows sampled. Check --sample-size and question files.")

    sample_counts: dict[str, int] = {}
    for rec in sampled_records:
        sample_counts[rec.question_type] = sample_counts.get(rec.question_type, 0) + 1

    sample_rows = [dict(rec.row, question_type=rec.question_type, source_file=rec.source_file) for rec in sampled_records]
    sample_csv = output_dir / "context_bundle_sample.csv"
    pd.DataFrame(sample_rows).to_csv(sample_csv, index=False)

    answers: list[dict[str, Any] | None] = [None] * len(sampled_records)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.answer_workers)) as pool:
        futures: dict[Any, tuple[int, QuestionRecord, list[EvidenceItem]]] = {}
        for idx, record in enumerate(sampled_records):
            question = _safe_text(record.row.get("question"))
            evidence_bundle = _choose_evidence_bundle(
                record,
                evidence_count=args.evidence_count,
                pool_by_file=pool_by_file,
                global_pool=global_pool,
            )
            future = pool.submit(
                _answer_question_with_evidence,
                idx=idx,
                question=question,
                evidence_items=evidence_bundle,
                model=args.answer_model,
            )
            futures[future] = (idx, record, evidence_bundle)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating baseline answers"):
            idx, record, evidence_bundle = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                fallback = AnswerResult(
                    idx=idx,
                    direct_answer="",
                    reasoning_answer=f"[ERROR] {exc}",
                    system_answer=f"[ERROR] {exc}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    processing_time=0.0,
                    evidence_items=evidence_bundle,
                )
                answers[fallback.idx] = _build_output_row(record, fallback)
                continue
            answers[result.idx] = _build_output_row(record, result)
    answer_seconds = time.perf_counter() - started

    answer_rows = [row for row in answers if row is not None]
    answer_csv = output_dir / "context_bundle_sample_answers.csv"
    pd.DataFrame(answer_rows).to_csv(answer_csv, index=False)

    eval_csv = output_dir / "context_bundle_sample_eval.csv"
    eval_summary_json = Path(f"{eval_csv}.summary.json")
    eval_grouped_json = Path(f"{eval_csv}.grouped.summary.json")

    eval_started = time.perf_counter()
    evaluate_file(
        input_path=str(answer_csv),
        output_path=str(eval_csv),
        summary_path=str(eval_summary_json),
        group_summary_path=str(eval_grouped_json),
        pubmed_table_map_path=str(Path(args.pubmed_table_map).expanduser().resolve()),
        model=args.eval_model,
        skip_llm=args.skip_llm_eval,
        prompt_price=args.prompt_price,
        completion_price=args.completion_price,
        infer_tables_from_pubmed=args.infer_tables_from_pubmed,
        workers=args.eval_workers,
    )
    eval_seconds = time.perf_counter() - eval_started

    eval_summary = {}
    if eval_summary_json.exists():
        eval_summary = json.loads(eval_summary_json.read_text(encoding="utf-8"))

    script_summary = {
        "sample_size": len(sampled_records),
        "sample_counts_by_question_type": sample_counts,
        "evidence_count": args.evidence_count,
        "answer_model": args.answer_model,
        "eval_model": args.eval_model,
        "answer_seconds": answer_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": answer_seconds + eval_seconds,
        "files": {
            "sample_csv": str(sample_csv),
            "answer_csv": str(answer_csv),
            "eval_csv": str(eval_csv),
            "eval_summary_json": str(eval_summary_json),
            "eval_grouped_json": str(eval_grouped_json),
        },
        "evaluation_summary": eval_summary,
    }
    script_summary_path = output_dir / "context_bundle_run_summary.json"
    script_summary_path.write_text(json.dumps(script_summary, indent=2), encoding="utf-8")
    return script_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample questions from each type, answer using a fixed evidence bundle with guaranteed "
            "ground-truth evidence, then evaluate with the same LLM-judge pipeline as RAG."
        )
    )
    parser.add_argument("--questions-dir", default=str(DEFAULT_QUESTIONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pubmed-table-map", default=str(DEFAULT_PUBMED_TABLE_MAP))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--evidence-count", type=int, default=DEFAULT_EVIDENCE_COUNT)
    parser.add_argument("--max-evidence-chars", type=int, default=DEFAULT_MAX_EVIDENCE_CHARS)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--answer-model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--answer-workers", type=int, default=6)
    parser.add_argument("--eval-workers", type=int, default=10)
    parser.add_argument("--skip-llm-eval", action="store_true")
    parser.add_argument("--infer-tables-from-pubmed", action="store_true")
    parser.add_argument("--prompt-price", type=float, default=0.0)
    parser.add_argument("--completion-price", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()
    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. Set it in env or data_decomposer/.env before running."
        )

    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
