#!/usr/bin/env python3
"""Evaluate DrugBank RAG answer CSVs with lexical/source metrics and optional LLM judging."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from rouge_score import rouge_scorer
from tqdm import tqdm

from table_metadata_index import normalize_table_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PUBMED_TABLE_MAP = REPO_ROOT / "data" / "Pharma" / "pubmed-drugbank-tables.gt"
DEFAULT_MODEL = "openai/gpt-5"
ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
GPT5_CHAT_FALLBACK_MODEL = "openai/gpt-5-chat"
DEFAULT_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # Source: OpenRouter model card pricing (input/output) for openai/gpt-5 family.
    "openai/gpt-5": (1.25, 10.0),
    "openai/gpt-5-chat": (1.25, 10.0),
}


@dataclass
class OpenRouterConfig:
    client: OpenAI
    headers: dict[str, str]


@dataclass
class LLMScoreResult:
    score: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    processing_time: float


@dataclass
class LLMPairScoreResult:
    direct_score: float
    reasoning_score: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    processing_time: float
    api_calls: int


_THREAD_LOCAL = threading.local()


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


def _parse_float_from_text(text: str) -> float:
    match = re.search(r"\d*\.?\d+", text)
    if not match:
        raise ValueError(f"Could not parse score from LLM response: {text}")
    value = float(match.group())
    return max(0.0, min(1.0, value))


def _coerce_allowed_score(value: object) -> float:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        try:
            numeric = _parse_float_from_text(str(value))
        except Exception:
            return math.nan
    numeric = max(0.0, min(1.0, numeric))
    return min((0.0, 0.5, 1.0), key=lambda candidate: abs(candidate - numeric))


def _extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _estimate_tokens_from_text(text: str) -> int:
    # Practical approximation for English text when provider usage is unavailable.
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _estimate_pair_prompt_tokens(
    reference_direct: str,
    candidate_direct: str,
    reference_reasoning: str,
    candidate_reasoning: str,
) -> int:
    fixed_instruction_tokens = 120
    payload = "\n".join(
        [
            reference_direct or "",
            candidate_direct or "",
            reference_reasoning or "",
            candidate_reasoning or "",
        ]
    )
    return fixed_instruction_tokens + _estimate_tokens_from_text(payload)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if not content:
        return ""
    return str(content).strip()


def _chat_completion_with_gpt5_fallback(
    *,
    config: OpenRouterConfig,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[Any, str, float]:
    start_time = time.perf_counter()
    response = config.client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=config.headers,
    )
    duration = time.perf_counter() - start_time
    text = _response_text(response)
    if text or model != DEFAULT_MODEL:
        return response, text, duration

    fallback_start = time.perf_counter()
    fallback_response = config.client.chat.completions.create(
        model=GPT5_CHAT_FALLBACK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=config.headers,
    )
    fallback_duration = time.perf_counter() - fallback_start
    fallback_text = _response_text(fallback_response)
    return fallback_response, fallback_text, duration + fallback_duration


def score_answer(
    reference: str,
    candidate: str,
    config: OpenRouterConfig,
    model: str = DEFAULT_MODEL,
) -> LLMScoreResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grader. Given a reference answer and a model answer, "
                "respond with a score of either 0, 0.5, or 1 indicating how correct the model answer "
                "is compared to the reference. Return only the numeric score."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference answer:\n{reference}\n\n"
                f"Model answer:\n{candidate}\n\nScore:"
            ),
        },
    ]

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            response, text, duration = _chat_completion_with_gpt5_fallback(
                config=config,
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=20,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter scoring failed with no response")

    score = _parse_float_from_text(text)

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    return LLMScoreResult(
        score=score,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        processing_time=duration,
    )


def score_answer_pair(
    reference_direct: str,
    candidate_direct: str,
    reference_reasoning: str,
    candidate_reasoning: str,
    config: OpenRouterConfig,
    model: str = DEFAULT_MODEL,
) -> LLMPairScoreResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grader. Evaluate two answer pairs and return JSON only with keys "
                "'direct_score' and 'reasoning_score'. Allowed values are 0, 0.5, 1, or null when a pair is missing."
            ),
        },
        {
            "role": "user",
            "content": (
                "Direct reference:\n"
                f"{reference_direct or 'N/A'}\n\n"
                "Direct model answer:\n"
                f"{candidate_direct or 'N/A'}\n\n"
                "Reasoning reference:\n"
                f"{reference_reasoning or 'N/A'}\n\n"
                "Reasoning model answer:\n"
                f"{candidate_reasoning or 'N/A'}\n\n"
                'Return JSON: {"direct_score": <0|0.5|1|null>, "reasoning_score": <0|0.5|1|null>}'
            ),
        },
    ]

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            response, text, duration = _chat_completion_with_gpt5_fallback(
                config=config,
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=120,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter pair scoring failed with no response")

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    direct_score = math.nan
    reasoning_score = math.nan

    json_text = _extract_json_object(text)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                direct_score = _coerce_allowed_score(parsed.get("direct_score"))
                reasoning_score = _coerce_allowed_score(parsed.get("reasoning_score"))
        except json.JSONDecodeError:
            pass

    # Fallback: parse first two numeric values from free-form output.
    if math.isnan(direct_score) or math.isnan(reasoning_score):
        numbers = re.findall(r"\d*\.?\d+", text)
        if math.isnan(direct_score) and numbers:
            direct_score = _coerce_allowed_score(numbers[0])
        if math.isnan(reasoning_score) and len(numbers) > 1:
            reasoning_score = _coerce_allowed_score(numbers[1])

    api_calls = 1
    if math.isnan(direct_score) and reference_direct and candidate_direct:
        direct_result = score_answer(reference_direct, candidate_direct, config, model=model)
        direct_score = direct_result.score
        prompt_tokens += direct_result.prompt_tokens
        completion_tokens += direct_result.completion_tokens
        total_tokens += direct_result.total_tokens
        duration += direct_result.processing_time
        api_calls += 1
    if math.isnan(reasoning_score) and reference_reasoning and candidate_reasoning:
        reasoning_result = score_answer(reference_reasoning, candidate_reasoning, config, model=model)
        reasoning_score = reasoning_result.score
        prompt_tokens += reasoning_result.prompt_tokens
        completion_tokens += reasoning_result.completion_tokens
        total_tokens += reasoning_result.total_tokens
        duration += reasoning_result.processing_time
        api_calls += 1

    return LLMPairScoreResult(
        direct_score=direct_score,
        reasoning_score=reasoning_score,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        processing_time=duration,
        api_calls=api_calls,
    )


def calculate_rouge(reference: str, prediction: str) -> dict[str, float]:
    scores = ROUGE_SCORER.score(reference, prediction)
    return {name: float(value.fmeasure) for name, value in scores.items()}


def calculate_string_similarity(reference: str, prediction: str) -> float:
    return SequenceMatcher(None, reference, prediction).ratio()


def _normalise_source_name(source: str) -> str:
    text = str(source).strip()
    basename = os.path.splitext(os.path.basename(text))[0]
    normalised = basename.replace("-", "_").replace(" ", "_")
    return normalised.lower()


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _parse_list_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    if "," in text:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return [text]


def _normalise_table_names(values: Iterable[str]) -> list[str]:
    return [normalize_table_name(value) for value in values if str(value).strip()]


def calculate_mrr(system_sources: Iterable[str], gt_sources: Iterable[str]) -> float:
    if not system_sources:
        return 0.0
    gt = {_normalise_source_name(source) for source in gt_sources if source}
    if not gt:
        return 0.0
    for rank, source in enumerate(system_sources, start=1):
        if _normalise_source_name(source) in gt:
            return 1.0 / float(rank)
    return 0.0


def calculate_source_metrics(
    system_sources: Iterable[str],
    gt_sources: Iterable[str],
) -> dict[str, float]:
    system_set = {_normalise_source_name(source) for source in system_sources if source}
    gt_set = {_normalise_source_name(source) for source in gt_sources if source}
    if not system_set and not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_positives = len(system_set.intersection(gt_set))
    precision = true_positives / len(system_set) if system_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _compute_cost(prompt_tokens: int, completion_tokens: int, prompt_price: float, completion_price: float) -> float:
    if prompt_price <= 0 and completion_price <= 0:
        return 0.0
    return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price


def _resolve_model_prices(prompt_price: float, completion_price: float, model: str) -> tuple[float, float]:
    if prompt_price > 0 or completion_price > 0:
        return prompt_price, completion_price

    env_prompt = os.getenv("OPENROUTER_PROMPT_PRICE_PER_1K")
    env_completion = os.getenv("OPENROUTER_COMPLETION_PRICE_PER_1K")
    if env_prompt and env_completion:
        try:
            return float(env_prompt), float(env_completion)
        except ValueError:
            pass

    direct = DEFAULT_PRICING_PER_1M.get(model)
    if direct is not None:
        return direct[0] / 1000.0, direct[1] / 1000.0

    if model.startswith("openai/gpt-5"):
        # Keep this family fallback for variant IDs like openai/gpt-5-<date>.
        base = DEFAULT_PRICING_PER_1M["openai/gpt-5"]
        return base[0] / 1000.0, base[1] / 1000.0

    return 0.0, 0.0


def _load_pubmed_table_mapping(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "," not in line:
                continue
            pubmed_id, table_file = line.split(",", 1)
            table_name = normalize_table_name(Path(table_file.strip()).stem)
            mapping.setdefault(pubmed_id.strip(), set()).add(table_name)
    return {k: sorted(v) for k, v in mapping.items()}


def _infer_question_family(question_id: str) -> str:
    if not question_id:
        return "unknown"
    match = re.match(r"([a-zA-Z]+)", question_id)
    if not match:
        return "unknown"
    return match.group(1).lower()


def evaluate_file(
    input_path: str,
    output_path: str,
    summary_path: str,
    group_summary_path: str,
    pubmed_table_map_path: str,
    model: str,
    skip_llm: bool,
    prompt_price: float,
    completion_price: float,
    infer_tables_from_pubmed: bool,
    workers: int,
) -> None:
    prompt_price, completion_price = _resolve_model_prices(prompt_price, completion_price, model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(group_summary_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    pubmed_table_map = _load_pubmed_table_mapping(Path(pubmed_table_map_path))

    def _evaluate_row(index: int, row_obj: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any], int, int, float, float]:
        reference_direct = _safe_text(row_obj.get("short_answer")) or _safe_text(row_obj.get("answer_direct"))
        reference_reasoning = _safe_text(row_obj.get("answer_reasoning"))
        reference_full = "\n\n".join(part for part in [reference_direct, reference_reasoning] if part)

        direct_system_answer = _safe_text(row_obj.get("direct_system_answer"))
        reasoning_system_answer = _safe_text(row_obj.get("reasoning_system_answer"))
        system_answer = _safe_text(row_obj.get("system_answer"))

        direct_metrics = calculate_rouge(reference_direct, direct_system_answer) if reference_direct and direct_system_answer else {}
        reasoning_metrics = calculate_rouge(reference_reasoning, reasoning_system_answer) if reference_reasoning and reasoning_system_answer else {}
        full_metrics = calculate_rouge(reference_full, system_answer) if reference_full and system_answer else {}

        direct_similarity = (
            calculate_string_similarity(reference_direct, direct_system_answer)
            if reference_direct and direct_system_answer
            else math.nan
        )
        reasoning_similarity = (
            calculate_string_similarity(reference_reasoning, reasoning_system_answer)
            if reference_reasoning and reasoning_system_answer
            else math.nan
        )
        full_similarity = (
            calculate_string_similarity(reference_full, system_answer)
            if reference_full and system_answer
            else math.nan
        )

        llm_direct_score = math.nan
        llm_reasoning_score = math.nan
        llm_prompt_tokens = 0
        llm_completion_tokens = 0
        llm_total_tokens = 0
        llm_processing_time = 0.0
        llm_calls = 0

        if not skip_llm:
            openrouter = _get_thread_openrouter_config()
            if (reference_direct and direct_system_answer) or (reference_reasoning and reasoning_system_answer):
                try:
                    pair_result = score_answer_pair(
                        reference_direct=reference_direct,
                        candidate_direct=direct_system_answer,
                        reference_reasoning=reference_reasoning,
                        candidate_reasoning=reasoning_system_answer,
                        config=openrouter,
                        model=model,
                    )
                    llm_direct_score = pair_result.direct_score
                    llm_reasoning_score = pair_result.reasoning_score
                    llm_prompt_tokens += pair_result.prompt_tokens
                    llm_completion_tokens += pair_result.completion_tokens
                    llm_total_tokens += pair_result.total_tokens
                    llm_processing_time += pair_result.processing_time
                    llm_calls += pair_result.api_calls
                except Exception:
                    # Keep lexical/source metrics even when external LLM judging fails.
                    llm_direct_score = math.nan
                    llm_reasoning_score = math.nan
                    estimated_prompt_tokens = _estimate_pair_prompt_tokens(
                        reference_direct,
                        direct_system_answer,
                        reference_reasoning,
                        reasoning_system_answer,
                    )
                    estimated_completion_tokens = 16
                    llm_prompt_tokens += estimated_prompt_tokens
                    llm_completion_tokens += estimated_completion_tokens
                    llm_total_tokens += estimated_prompt_tokens + estimated_completion_tokens
                    llm_calls += 1

        llm_avg_score = (
            float(pd.Series([llm_direct_score, llm_reasoning_score]).dropna().mean())
            if not math.isnan(llm_direct_score) or not math.isnan(llm_reasoning_score)
            else math.nan
        )

        gt_passages = _parse_list_field(row_obj.get("pubmed_id"))
        system_passages = _parse_list_field(row_obj.get("passages_used"))
        passage_metrics = calculate_source_metrics(system_passages, gt_passages)
        passage_mrr = calculate_mrr(system_passages, gt_passages)

        explicit_gt_tables = _normalise_table_names(_parse_list_field(row_obj.get("table_id")))
        if explicit_gt_tables:
            gt_tables = sorted(set(explicit_gt_tables))
        elif infer_tables_from_pubmed:
            gt_tables_set: set[str] = set()
            for pubmed_id in gt_passages:
                for table_name in pubmed_table_map.get(pubmed_id, []):
                    gt_tables_set.add(table_name)
            gt_tables = sorted(gt_tables_set)
        else:
            gt_tables = []
        system_tables = _normalise_table_names(_parse_list_field(row_obj.get("tables_used")))
        table_metrics = calculate_source_metrics(system_tables, gt_tables)
        table_mrr = calculate_mrr(system_tables, gt_tables)

        row_cost = _compute_cost(
            llm_prompt_tokens,
            llm_completion_tokens,
            prompt_price=prompt_price,
            completion_price=completion_price,
        )

        metrics_row = {
            "direct_rouge1": direct_metrics.get("rouge1", math.nan),
            "direct_rouge2": direct_metrics.get("rouge2", math.nan),
            "direct_rougeL": direct_metrics.get("rougeL", math.nan),
            "direct_string_similarity": direct_similarity,
            "reasoning_rouge1": reasoning_metrics.get("rouge1", math.nan),
            "reasoning_rouge2": reasoning_metrics.get("rouge2", math.nan),
            "reasoning_rougeL": reasoning_metrics.get("rougeL", math.nan),
            "reasoning_string_similarity": reasoning_similarity,
            "full_rouge1": full_metrics.get("rouge1", math.nan),
            "full_rouge2": full_metrics.get("rouge2", math.nan),
            "full_rougeL": full_metrics.get("rougeL", math.nan),
            "full_string_similarity": full_similarity,
            "passage_precision": passage_metrics["precision"],
            "passage_recall": passage_metrics["recall"],
            "passage_f1": passage_metrics["f1"],
            "passage_mrr": passage_mrr,
            "table_precision": table_metrics["precision"],
            "table_recall": table_metrics["recall"],
            "table_f1": table_metrics["f1"],
            "table_mrr": table_mrr,
            "llm_direct_correctness": llm_direct_score,
            "llm_reasoning_correctness": llm_reasoning_score,
            "llm_avg_correctness": llm_avg_score,
            "llm_prompt_tokens": llm_prompt_tokens,
            "llm_completion_tokens": llm_completion_tokens,
            "llm_total_tokens": llm_total_tokens,
            "llm_processing_time": llm_processing_time,
            "llm_inference_cost": row_cost,
        }
        combined_row = {**row_obj, **metrics_row}
        return index, metrics_row, combined_row, llm_total_tokens, llm_calls, row_cost, llm_processing_time

    records = df.to_dict(orient="records")
    metrics_rows: list[dict[str, Any] | None] = [None] * len(records)
    combined_rows: list[dict[str, Any] | None] = [None] * len(records)
    aggregate_tokens = 0
    aggregate_calls = 0
    aggregate_cost = 0.0
    aggregate_times: list[float] = []

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_evaluate_row, i, row_obj) for i, row_obj in enumerate(records)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating rows"):
            idx, metrics_row, combined_row, llm_total_tokens, llm_calls, row_cost, llm_processing_time = future.result()
            metrics_rows[idx] = metrics_row
            combined_rows[idx] = combined_row
            aggregate_tokens += llm_total_tokens
            aggregate_calls += llm_calls
            aggregate_cost += row_cost
            if llm_processing_time:
                aggregate_times.append(llm_processing_time)

    elapsed = time.perf_counter() - started
    row_throughput = (len(records) / elapsed) if elapsed > 0 else float("inf")
    print(f"Evaluated {len(records)} rows in {elapsed:.2f}s ({row_throughput:.2f} rows/s)")
    if not skip_llm:
        llm_call_throughput = (aggregate_calls / elapsed) if elapsed > 0 else float("inf")
        print(f"LLM scoring calls: {aggregate_calls} ({llm_call_throughput:.2f} calls/s)")

    metrics_df = pd.DataFrame([row for row in metrics_rows if row is not None])
    combined_df = pd.DataFrame([row for row in combined_rows if row is not None])
    combined_df.to_csv(output_path, index=False)


    summary = {
        "row_count": int(len(combined_df)),
        "average_metrics": metrics_df.mean(numeric_only=True, skipna=True).to_dict(),
        "total_llm_tokens": int(aggregate_tokens),
        "total_llm_api_calls": int(aggregate_calls),
        "total_llm_inference_cost": float(aggregate_cost),
        "average_llm_processing_time": float(sum(aggregate_times) / len(aggregate_times)) if aggregate_times else 0.0,
        "pricing": {
            "prompt_price_per_1k": float(prompt_price),
            "completion_price_per_1k": float(completion_price),
            "model": model,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if "question_id" in combined_df.columns:
        combined_df["question_family"] = combined_df["question_id"].astype(str).map(_infer_question_family)
        grouped = combined_df.groupby("question_family", dropna=False).mean(numeric_only=True).to_dict(orient="index")
    else:
        grouped = {}
    with open(group_summary_path, "w", encoding="utf-8") as handle:
        json.dump(grouped, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DrugBank RAG answers stored in CSVs.")
    parser.add_argument("input_csv", help="Path to answered CSV (rag_answer_questions.py output)")
    parser.add_argument("output_csv", help="Path to write CSV with metrics")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for summary JSON (defaults to <output_csv>.summary.json)",
    )
    parser.add_argument(
        "--pubmed-table-map",
        default=str(DEFAULT_PUBMED_TABLE_MAP),
        help="Path to pubmed->table mapping file (.gt).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model for LLM judging")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM judging (metrics only)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel evaluator workers.")
    parser.add_argument(
        "--infer-tables-from-pubmed",
        action="store_true",
        help=(
            "Infer table ground truth from pubmed_id mapping when table_id is missing. "
            "Disabled by default for strict passage-only evaluation."
        ),
    )
    parser.add_argument(
        "--prompt-price",
        type=float,
        default=0.0,
        help="USD price per 1K prompt tokens (optional for cost estimation)",
    )
    parser.add_argument(
        "--completion-price",
        type=float,
        default=0.0,
        help="USD price per 1K completion tokens (optional for cost estimation)",
    )
    args = parser.parse_args()

    load_dotenv()
    summary_path = args.summary_json or f"{args.output_csv}.summary.json"
    group_summary_path = f"{args.output_csv}.grouped.summary.json"

    if not args.skip_llm and not os.getenv("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")

    evaluate_file(
        input_path=args.input_csv,
        output_path=args.output_csv,
        summary_path=summary_path,
        group_summary_path=group_summary_path,
        pubmed_table_map_path=args.pubmed_table_map,
        model=args.model,
        skip_llm=args.skip_llm,
        prompt_price=args.prompt_price,
        completion_price=args.completion_price,
        infer_tables_from_pubmed=args.infer_tables_from_pubmed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
