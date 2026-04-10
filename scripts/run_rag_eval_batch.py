#!/usr/bin/env python3
"""Run DrugBank RAG + evaluation over multiple question files in parallel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass
class TaskResult:
    question_file: str
    answer_csv: str
    eval_csv: str
    row_count: int
    status: str
    error: str
    full_rouge1: float
    full_rouge2: float
    full_rougeL: float
    passage_mrr: float
    table_mrr: float
    llm_avg_correctness: float
    llm_avg_inference_cost: float
    llm_total_inference_cost: float
    rag_seconds: float
    eval_seconds: float
    total_seconds: float
    rag_qps: float
    eval_rps: float


def _detect_gpu_ids() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    ids: list[int] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids


def _run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(cmd, env=env, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            "See command output above for details."
        )


def _safe_mean_dict(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    available = [col for col in columns if col in df.columns]
    if not available or df.empty:
        return {}
    return df[available].mean(numeric_only=True, skipna=True).to_dict()


def _build_summary_payload(
    *,
    question_files: list[Path],
    results: list[TaskResult],
    failures: list[TaskResult],
    elapsed_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [asdict(result) for result in results + failures]
    results_df = pd.DataFrame(rows)
    avg_metrics = _safe_mean_dict(
        results_df,
        [
            "full_rouge1",
            "full_rouge2",
            "full_rougeL",
            "passage_mrr",
            "table_mrr",
            "llm_avg_correctness",
            "llm_avg_inference_cost",
        ],
    )
    speed_metrics = _safe_mean_dict(
        results_df,
        ["row_count", "rag_seconds", "eval_seconds", "total_seconds", "rag_qps", "eval_rps"],
    )
    total_llm_cost = (
        float(results_df["llm_total_inference_cost"].sum(skipna=True))
        if "llm_total_inference_cost" in results_df
        else 0.0
    )
    completed = len(results) + len(failures)
    pending_files = [path.name for path in question_files]
    done_files = {result.question_file for result in results + failures}
    pending_files = [name for name in pending_files if name not in done_files]
    payload = {
        "total_files": len(question_files),
        "completed_files": completed,
        "success_count": int(len(results)),
        "failure_count": int(len(failures)),
        "pending_files": pending_files,
        "elapsed_seconds": elapsed_seconds,
        "average_metrics": avg_metrics,
        "total_llm_inference_cost": total_llm_cost,
        "speed": speed_metrics,
        "files": rows,
    }
    return results_df, payload


def _write_progress_artifacts(
    *,
    output_dir: Path,
    question_files: list[Path],
    results: list[TaskResult],
    failures: list[TaskResult],
    started_at: float,
) -> None:
    elapsed_seconds = time.perf_counter() - started_at
    results_df, payload = _build_summary_payload(
        question_files=question_files,
        results=results,
        failures=failures,
        elapsed_seconds=elapsed_seconds,
    )
    payload["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    progress_csv = output_dir / "batch_progress_metrics.csv"
    progress_json = output_dir / "batch_progress_status.json"
    results_df.to_csv(progress_csv, index=False)
    progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _worker_run(
    *,
    question_file: Path,
    output_dir: Path,
    answer_script: Path,
    db_path: Path,
    table_index_name: str,
    passage_index_name: str,
    pubmed_table_map: Path,
    limit_per_file: int,
    python_executable: str,
    nl2sql_model: str,
    answer_model: str,
    eval_model: str,
    skip_llm_eval: bool,
    gpu_id: Optional[int],
    question_workers: int,
    eval_workers: int,
    embed_batch_size: int,
    save_every: int,
    prompt_price: float | None,
    completion_price: float | None,
) -> TaskResult:
    stem = question_file.stem
    answer_csv = output_dir / f"{stem}_rag_sample.csv"
    eval_csv = output_dir / f"{stem}_rag_sample_eval.csv"

    env = os.environ.copy()
    device = "auto"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"

    rag_cmd = [
        python_executable,
        str(answer_script),
        "--input-csv",
        str(question_file),
        "--output-csv",
        str(answer_csv),
        "--db-path",
        str(db_path),
        "--pubmed-table-map",
        str(pubmed_table_map),
        "--table-index-name",
        table_index_name,
        "--passage-index-name",
        passage_index_name,
        "--device",
        device,
        "--nl2sql-model",
        nl2sql_model,
        "--answer-model",
        answer_model,
        "--question-workers",
        str(question_workers),
        "--embed-batch-size",
        str(embed_batch_size),
        "--save-every",
        str(save_every),
    ]
    if limit_per_file > 0:
        rag_cmd.extend(["--limit", str(limit_per_file)])
    rag_started = time.perf_counter()
    _run_cmd(rag_cmd, env=env)
    rag_seconds = time.perf_counter() - rag_started

    eval_cmd = [
        python_executable,
        str(SCRIPT_DIR / "evaluate_rag_answers.py"),
        str(answer_csv),
        str(eval_csv),
        "--pubmed-table-map",
        str(pubmed_table_map),
        "--model",
        eval_model,
        "--workers",
        str(eval_workers),
    ]
    if prompt_price is not None:
        eval_cmd.extend(["--prompt-price", str(prompt_price)])
    if completion_price is not None:
        eval_cmd.extend(["--completion-price", str(completion_price)])
    if skip_llm_eval:
        eval_cmd.append("--skip-llm")
    eval_started = time.perf_counter()
    _run_cmd(eval_cmd, env=env)
    eval_seconds = time.perf_counter() - eval_started

    summary_path = Path(f"{eval_csv}.summary.json")
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    avg_metrics = summary.get("average_metrics", {}) if isinstance(summary, dict) else {}
    if not isinstance(avg_metrics, dict):
        avg_metrics = {}
    total_llm_cost = summary.get("total_llm_inference_cost", 0.0) if isinstance(summary, dict) else 0.0
    try:
        total_llm_cost_value = float(total_llm_cost)
    except (TypeError, ValueError):
        total_llm_cost_value = float("nan")

    row_count = 0
    if answer_csv.exists():
        row_count = len(pd.read_csv(answer_csv))
    rag_qps = (row_count / rag_seconds) if rag_seconds > 0 else float("nan")
    eval_rps = (row_count / eval_seconds) if eval_seconds > 0 else float("nan")

    def _metric(name: str) -> float:
        value = avg_metrics.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return TaskResult(
        question_file=question_file.name,
        answer_csv=str(answer_csv),
        eval_csv=str(eval_csv),
        row_count=row_count,
        status="ok",
        error="",
        full_rouge1=_metric("full_rouge1"),
        full_rouge2=_metric("full_rouge2"),
        full_rougeL=_metric("full_rougeL"),
        passage_mrr=_metric("passage_mrr"),
        table_mrr=_metric("table_mrr"),
        llm_avg_correctness=_metric("llm_avg_correctness"),
        llm_avg_inference_cost=_metric("llm_inference_cost"),
        llm_total_inference_cost=total_llm_cost_value,
        rag_seconds=rag_seconds,
        eval_seconds=eval_seconds,
        total_seconds=rag_seconds + eval_seconds,
        rag_qps=rag_qps,
        eval_rps=eval_rps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sample RAG + evaluation for all question types in parallel."
    )
    parser.add_argument("--questions-dir", default=str(REPO_ROOT / "questions_final"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "questions_final_test" / "batch_samples"))
    parser.add_argument(
        "--answer-script",
        default=str(SCRIPT_DIR / "rag_answer_questions.py"),
        help="Path to answer-generation script (default: rag_answer_questions.py).",
    )
    parser.add_argument("--db-path", default=str(REPO_ROOT / "data" / "drugbank.db"))
    parser.add_argument("--pubmed-table-map", default=str(REPO_ROOT / "data" / "Pharma" / "pubmed-drugbank-tables.gt"))
    parser.add_argument("--table-index-name", default="drug_bank_data_lake_tables")
    parser.add_argument("--passage-index-name", default="drug_bank_data_lake")
    parser.add_argument("--limit-per-file", type=int, default=0, help="0 = process all rows in each file")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--question-workers", type=int, default=16, help="Per-file parallel workers for RAG generation.")
    parser.add_argument("--eval-workers", type=int, default=24, help="Per-file parallel workers for evaluation.")
    parser.add_argument("--embed-batch-size", type=int, default=256, help="Per-file embedding batch size.")
    parser.add_argument("--save-every", type=int, default=100, help="Per-file incremental CSV save cadence.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--nl2sql-model", default="openai/gpt-5")
    parser.add_argument("--answer-model", default="openai/gpt-5")
    parser.add_argument("--eval-model", default="openai/gpt-5")
    parser.add_argument("--skip-llm-eval", action="store_true")
    parser.add_argument("--prompt-price", type=float, default=None, help="USD per 1K prompt tokens for evaluator cost.")
    parser.add_argument(
        "--completion-price",
        type=float,
        default=None,
        help="USD per 1K completion tokens for evaluator cost.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    questions_dir = Path(args.questions_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    answer_script = Path(args.answer_script).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    pubmed_table_map = Path(args.pubmed_table_map).expanduser().resolve()

    if not answer_script.exists():
        print(f"Answer script not found: {answer_script}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    question_files = sorted(questions_dir.glob("*_processed.csv"))
    if args.max_files is not None:
        question_files = question_files[: args.max_files]
    if not question_files:
        print(f"No *_processed.csv files found under {questions_dir}")
        return 1

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = _detect_gpu_ids()
    if not gpu_ids:
        gpu_ids = [None]  # CPU / default device

    results: list[TaskResult] = []
    failures: list[TaskResult] = []
    started_at = time.perf_counter()
    _write_progress_artifacts(
        output_dir=output_dir,
        question_files=question_files,
        results=results,
        failures=failures,
        started_at=started_at,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures: dict[Future[TaskResult], Path] = {}
        for idx, question_file in enumerate(question_files):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            future = pool.submit(
                _worker_run,
                question_file=question_file,
                output_dir=output_dir,
                answer_script=answer_script,
                db_path=db_path,
                table_index_name=args.table_index_name,
                passage_index_name=args.passage_index_name,
                pubmed_table_map=pubmed_table_map,
                limit_per_file=args.limit_per_file,
                python_executable=args.python_executable,
                nl2sql_model=args.nl2sql_model,
                answer_model=args.answer_model,
                eval_model=args.eval_model,
                skip_llm_eval=args.skip_llm_eval,
                gpu_id=gpu_id if isinstance(gpu_id, int) else None,
                question_workers=args.question_workers,
                eval_workers=args.eval_workers,
                embed_batch_size=args.embed_batch_size,
                save_every=args.save_every,
                prompt_price=args.prompt_price,
                completion_price=args.completion_price,
            )
            futures[future] = question_file
        for future in as_completed(futures):
            question_file = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    "[ok] "
                    f"{result.question_file} rows={result.row_count} "
                    f"rag_qps={result.rag_qps:.2f} eval_rps={result.eval_rps:.2f} "
                    f"-> {result.eval_csv}",
                    flush=True,
                )
            except Exception as exc:
                failed = TaskResult(
                    question_file=question_file.name,
                    answer_csv="",
                    eval_csv="",
                    row_count=0,
                    status="failed",
                    error=str(exc),
                    full_rouge1=float("nan"),
                    full_rouge2=float("nan"),
                    full_rougeL=float("nan"),
                    passage_mrr=float("nan"),
                    table_mrr=float("nan"),
                    llm_avg_correctness=float("nan"),
                    llm_avg_inference_cost=float("nan"),
                    llm_total_inference_cost=float("nan"),
                    rag_seconds=float("nan"),
                    eval_seconds=float("nan"),
                    total_seconds=float("nan"),
                    rag_qps=float("nan"),
                    eval_rps=float("nan"),
                )
                failures.append(failed)
                print(f"[failed] {question_file.name}: {exc}", flush=True)

            completed_files = len(results) + len(failures)
            print(
                "[progress] "
                f"completed={completed_files}/{len(question_files)} "
                f"success={len(results)} failed={len(failures)}",
                flush=True,
            )
            _write_progress_artifacts(
                output_dir=output_dir,
                question_files=question_files,
                results=results,
                failures=failures,
                started_at=started_at,
            )

    results_df, summary_payload = _build_summary_payload(
        question_files=question_files,
        results=results,
        failures=failures,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    summary_csv = output_dir / "batch_sample_metrics_summary.csv"
    summary_json = output_dir / "batch_sample_metrics_summary.json"
    results_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
