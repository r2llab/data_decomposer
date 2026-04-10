#!/usr/bin/env python3
"""Run a zero-shot DrugBank baseline over all processed question CSVs.

This baseline does not retrieve any passages/tables. It sends only the raw
question to an LLM (via OpenRouter), writes answered rows to a combined CSV,
and can optionally run the existing evaluator.
"""

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
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from evaluate_rag_answers import evaluate_file


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_QUESTIONS_DIR = REPO_ROOT / "questions_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "questions_final_test" / "zero_shot_baseline"
DEFAULT_OUTPUT_CSV_NAME = "questions_final_zero_shot_answers.csv"
DEFAULT_SUMMARY_JSON_NAME = "zero_shot_run_summary.json"
DEFAULT_MODEL = "openai/gpt-5-chat"
DEFAULT_EVAL_MODEL = "openai/gpt-5-chat"
DEFAULT_MAX_TOKENS = 220
DEFAULT_PUBMED_TABLE_MAP = REPO_ROOT / "data" / "Pharma" / "pubmed-drugbank-tables.gt"

DEFAULT_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # OpenRouter model card pricing (input/output) for GPT-5 family.
    "openai/gpt-5": (1.25, 10.0),
    "openai/gpt-5-chat": (1.25, 10.0),
}

_THREAD_LOCAL = threading.local()


@dataclass
class OpenRouterConfig:
    client: OpenAI
    headers: dict[str, str]


@dataclass
class QuestionTask:
    idx: int
    question_type: str
    source_file: str
    row: dict[str, Any]


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
    error: str


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _question_type_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("_processed", "")


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
        base = DEFAULT_PRICING_PER_1M["openai/gpt-5"]
        return base[0] / 1000.0, base[1] / 1000.0

    return 0.0, 0.0


def _compute_cost(prompt_tokens: int, completion_tokens: int, prompt_price: float, completion_price: float) -> float:
    return (prompt_tokens / 1000.0) * prompt_price + (completion_tokens / 1000.0) * completion_price


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


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _answer_question(*, idx: int, question: str, model: str, max_tokens: int) -> AnswerResult:
    config = _get_thread_openrouter_config()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical QA assistant for DrugBank-style questions. "
                "You are given only a question and no additional evidence. "
                "Return strict JSON with keys direct_answer and reasoning_answer. "
                "direct_answer must be 1 concise sentence. "
                "reasoning_answer must be 2-3 short sentences and under 90 words. "
                "If uncertain, make your best effort and say uncertainty clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Return JSON only:\n"
                '{"direct_answer": "...", "reasoning_answer": "..."}'
            ),
        },
    ]

    def _request(max_tokens_override: int | None) -> tuple[Any, float]:
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "extra_headers": config.headers,
                }
                if max_tokens_override and max_tokens_override > 0:
                    kwargs["max_tokens"] = max_tokens_override
                started = time.perf_counter()
                response = config.client.chat.completions.create(**kwargs)
                duration = time.perf_counter() - started
                return response, duration
            except Exception as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 10))
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter answer generation failed with no response.")

    responses: list[Any] = []
    total_duration = 0.0

    response, duration = _request(max_tokens if max_tokens > 0 else None)
    responses.append(response)
    total_duration += duration
    content = (response.choices[0].message.content or "").strip()

    # Some reasoning-heavy endpoints can consume capped tokens without emitting
    # final visible text. Retry once without a cap so we still get an answer.
    if not content and max_tokens > 0:
        retry_response, retry_duration = _request(None)
        responses.append(retry_response)
        total_duration += retry_duration
        content = (retry_response.choices[0].message.content or "").strip()
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
        direct_answer = lines[0] if lines else "Unable to provide an answer."
        reasoning_answer = content or direct_answer

    if not direct_answer and reasoning_answer:
        direct_answer = reasoning_answer.split("\n", 1)[0].strip() or "Unable to provide an answer."
    if not reasoning_answer:
        reasoning_answer = "No additional reasoning provided."

    system_answer = f"{direct_answer}\n\n{reasoning_answer}".strip()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for resp in responses:
        usage = getattr(resp, "usage", None)
        p = int(getattr(usage, "prompt_tokens", 0) or 0)
        c = int(getattr(usage, "completion_tokens", 0) or 0)
        t = int(getattr(usage, "total_tokens", p + c) or 0)
        prompt_tokens += p
        completion_tokens += c
        total_tokens += t if t > 0 else (p + c)

    # Fallback estimate when provider usage is unavailable.
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _estimate_tokens_from_text(json.dumps(messages, ensure_ascii=False))
        completion_tokens = _estimate_tokens_from_text(content)
        total_tokens = prompt_tokens + completion_tokens
    elif total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return AnswerResult(
        idx=idx,
        direct_answer=direct_answer,
        reasoning_answer=reasoning_answer,
        system_answer=system_answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        processing_time=total_duration,
        error="",
    )


def _load_tasks(
    *,
    questions_dir: Path,
    limit_per_file: int,
    max_questions: int,
    max_files: int | None,
) -> list[QuestionTask]:
    question_files = sorted(questions_dir.glob("*_processed.csv"))
    if max_files is not None:
        question_files = question_files[:max_files]
    if not question_files:
        raise FileNotFoundError(f"No *_processed.csv files found in {questions_dir}")

    tasks: list[QuestionTask] = []
    idx = 0
    for csv_path in question_files:
        df = pd.read_csv(csv_path)
        if limit_per_file > 0:
            df = df.head(limit_per_file)
        q_type = _question_type_from_path(csv_path)
        source_file = csv_path.name
        for row in df.to_dict(orient="records"):
            tasks.append(
                QuestionTask(
                    idx=idx,
                    question_type=q_type,
                    source_file=source_file,
                    row=row,
                )
            )
            idx += 1
            if max_questions > 0 and len(tasks) >= max_questions:
                return tasks
    return tasks


def _build_output_row(task: QuestionTask, answer: AnswerResult, answer_model: str, answer_cost: float) -> dict[str, Any]:
    row = dict(task.row)
    row.update(
        {
            "question_type": task.question_type,
            "source_file": task.source_file,
            "system_answer": answer.system_answer,
            "direct_system_answer": answer.direct_answer,
            "reasoning_system_answer": answer.reasoning_answer,
            "passages_used": "",
            "tables_used": "",
            "answer_model": answer_model,
            "answer_prompt_tokens": answer.prompt_tokens,
            "answer_completion_tokens": answer.completion_tokens,
            "answer_total_tokens": answer.total_tokens,
            "answer_processing_time": answer.processing_time,
            "answer_inference_cost": answer_cost,
            "answer_error": answer.error,
        }
    )
    return row


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")

    questions_dir = Path(args.questions_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else (output_dir / DEFAULT_OUTPUT_CSV_NAME)
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else (output_dir / DEFAULT_SUMMARY_JSON_NAME)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    prompt_price, completion_price = _resolve_model_prices(
        prompt_price=args.prompt_price,
        completion_price=args.completion_price,
        model=args.answer_model,
    )

    tasks = _load_tasks(
        questions_dir=questions_dir,
        limit_per_file=args.limit_per_file,
        max_questions=args.max_questions,
        max_files=args.max_files,
    )
    if not tasks:
        raise RuntimeError("No question rows found after applying limits.")

    answers: list[dict[str, Any] | None] = [None] * len(tasks)
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "inference_cost": 0.0,
    }
    per_type_counts: dict[str, int] = {}
    for task in tasks:
        per_type_counts[task.question_type] = per_type_counts.get(task.question_type, 0) + 1

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.answer_workers)) as pool:
        futures: dict[Any, QuestionTask] = {}
        for task in tasks:
            question = _safe_text(task.row.get("question"))
            futures[
                pool.submit(
                    _answer_question,
                    idx=task.idx,
                    question=question,
                    model=args.answer_model,
                    max_tokens=args.max_tokens,
                )
            ] = task

        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Zero-shot answering"):
            task = futures[future]
            try:
                answer = future.result()
            except Exception as exc:
                answer = AnswerResult(
                    idx=task.idx,
                    direct_answer=f"[ERROR] {exc}",
                    reasoning_answer=f"[ERROR] {exc}",
                    system_answer=f"[ERROR] {exc}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    processing_time=0.0,
                    error=str(exc),
                )

            answer_cost = _compute_cost(
                answer.prompt_tokens,
                answer.completion_tokens,
                prompt_price=prompt_price,
                completion_price=completion_price,
            )
            answers[task.idx] = _build_output_row(task, answer, args.answer_model, answer_cost)

            totals["prompt_tokens"] += answer.prompt_tokens
            totals["completion_tokens"] += answer.completion_tokens
            totals["total_tokens"] += answer.total_tokens
            totals["inference_cost"] += answer_cost

            completed += 1
            if completed % max(1, args.save_every) == 0 or completed == len(tasks):
                pd.DataFrame([row for row in answers if row is not None]).to_csv(output_csv, index=False)

    elapsed = time.perf_counter() - started
    throughput = len(tasks) / elapsed if elapsed > 0 else float("inf")

    eval_summary: dict[str, Any] = {}
    eval_csv_path: Path | None = None
    eval_summary_path: Path | None = None
    eval_group_summary_path: Path | None = None
    if args.evaluate:
        eval_csv_path = Path(args.eval_csv).expanduser().resolve() if args.eval_csv else output_csv.with_name(
            f"{output_csv.stem}_eval.csv"
        )
        eval_summary_path = Path(f"{eval_csv_path}.summary.json")
        eval_group_summary_path = Path(f"{eval_csv_path}.grouped.summary.json")
        evaluate_file(
            input_path=str(output_csv),
            output_path=str(eval_csv_path),
            summary_path=str(eval_summary_path),
            group_summary_path=str(eval_group_summary_path),
            pubmed_table_map_path=str(Path(args.pubmed_table_map).expanduser().resolve()),
            model=args.eval_model,
            skip_llm=args.skip_llm_eval,
            prompt_price=args.eval_prompt_price,
            completion_price=args.eval_completion_price,
            infer_tables_from_pubmed=args.infer_tables_from_pubmed,
            workers=args.eval_workers,
        )
        if eval_summary_path.exists():
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "questions_dir": str(questions_dir),
        "output_csv": str(output_csv),
        "summary_json": str(summary_json),
        "row_count": len(tasks),
        "rows_per_type": per_type_counts,
        "answer_model": args.answer_model,
        "answer_workers": int(args.answer_workers),
        "answer_max_tokens": int(args.max_tokens),
        "answer_elapsed_seconds": elapsed,
        "answer_throughput_rows_per_second": throughput,
        "answer_prompt_tokens": int(totals["prompt_tokens"]),
        "answer_completion_tokens": int(totals["completion_tokens"]),
        "answer_total_tokens": int(totals["total_tokens"]),
        "answer_total_inference_cost": float(totals["inference_cost"]),
        "answer_average_inference_cost": float(totals["inference_cost"] / len(tasks)),
        "answer_pricing": {
            "prompt_price_per_1k": float(prompt_price),
            "completion_price_per_1k": float(completion_price),
        },
        "evaluation": {
            "enabled": bool(args.evaluate),
            "eval_csv": str(eval_csv_path) if eval_csv_path else "",
            "eval_summary_json": str(eval_summary_path) if eval_summary_path else "",
            "eval_grouped_summary_json": str(eval_group_summary_path) if eval_group_summary_path else "",
            "skip_llm_eval": bool(args.skip_llm_eval),
            "eval_model": args.eval_model,
            "summary": eval_summary,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zero-shot OpenRouter QA baseline over questions_final CSVs.")
    parser.add_argument("--questions-dir", default=str(DEFAULT_QUESTIONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-csv", default=None, help="Combined answered CSV path.")
    parser.add_argument("--summary-json", default=None, help="Run summary JSON path.")
    parser.add_argument("--answer-model", default=DEFAULT_MODEL, help="OpenRouter model for zero-shot answers.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens per answer.")
    parser.add_argument("--answer-workers", type=int, default=8, help="Parallel workers for answer generation.")
    parser.add_argument("--save-every", type=int, default=100, help="Persist intermediate CSV every N completed rows.")
    parser.add_argument("--limit-per-file", type=int, default=0, help="Cap rows per question file (0 = all).")
    parser.add_argument("--max-questions", type=int, default=0, help="Cap total rows across all files (0 = all).")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of input files.")
    parser.add_argument(
        "--prompt-price",
        type=float,
        default=0.0,
        help="USD price per 1K prompt tokens for answer generation (0 = auto/defaults).",
    )
    parser.add_argument(
        "--completion-price",
        type=float,
        default=0.0,
        help="USD price per 1K completion tokens for answer generation (0 = auto/defaults).",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run evaluator after generating answer CSV.")
    parser.add_argument("--eval-csv", default=None, help="Output CSV path for evaluated rows.")
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL, help="OpenRouter model for evaluator LLM grading.")
    parser.add_argument("--eval-workers", type=int, default=12, help="Parallel workers for evaluator.")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Skip evaluator LLM grading.")
    parser.add_argument("--infer-tables-from-pubmed", action="store_true", help="Enable evaluator table inference.")
    parser.add_argument("--pubmed-table-map", default=str(DEFAULT_PUBMED_TABLE_MAP))
    parser.add_argument(
        "--eval-prompt-price",
        type=float,
        default=0.0,
        help="USD price per 1K prompt tokens for evaluator (0 = auto/defaults).",
    )
    parser.add_argument(
        "--eval-completion-price",
        type=float,
        default=0.0,
        help="USD price per 1K completion tokens for evaluator (0 = auto/defaults).",
    )
    args = parser.parse_args()

    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
