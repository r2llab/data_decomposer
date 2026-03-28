"""End-to-end table QA pipeline with optional parallel processing.

This script performs:
1) Question generation over single tables.
2) Optional correctness filtering.
3) Optional paraphrasing.

Outputs are written to CSV files:
- generated file: all generated QA rows
- processed file: rows that pass optional processing stages
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from .openrouter_client import create_chat_completion, get_openrouter_client
    from .run_passage_pipeline import (
        DEFAULT_MODEL,
        FLUSH_EVERY,
        QuestionIdGenerator,
        RollingCsvWriter,
        calculate_llm_correctness,
        compose_reference_answer,
        extract_response_text,
        generate_llm_answer,
        paraphrase_question_item,
        parse_llm_response,
    )
except ImportError:
    from openrouter_client import create_chat_completion, get_openrouter_client
    from run_passage_pipeline import (
        DEFAULT_MODEL,
        FLUSH_EVERY,
        QuestionIdGenerator,
        RollingCsvWriter,
        calculate_llm_correctness,
        compose_reference_answer,
        extract_response_text,
        generate_llm_answer,
        paraphrase_question_item,
        parse_llm_response,
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DRUGBANK_TABLES_DIR = BASE_DIR.parent / "data" / "Pharma" / "drugbank-tables"

DEFAULT_GENERATED = BASE_DIR / "table_generated.csv"
DEFAULT_PROCESSED = BASE_DIR / "table_processed.csv"

TABLE_FIELDNAMES = [
    "question_id",
    "question",
    "short_answer",
    "answer_reasoning",
    "table_content",
    "table_id",
]


def to_table_row(row: Dict[str, str]) -> Dict[str, str]:
    """Map generic parsed row fields to table-output schema."""
    return {
        "question_id": row.get("question_id", ""),
        "question": row.get("question", ""),
        "short_answer": row.get("short_answer", ""),
        "answer_reasoning": row.get("answer_reasoning", ""),
        "table_content": row.get("table_content", row.get("pubmed_text", "")),
        "table_id": row.get("table_id", row.get("pubmed_id", "")),
    }


def load_drugbank_tables(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if not tables_dir.exists():
        raise FileNotFoundError(f"DrugBank tables directory does not exist: {tables_dir}")

    for file_path in sorted(tables_dir.glob("*.csv")):
        try:
            tables[file_path.stem] = pd.read_csv(file_path)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping unreadable table {file_path.name}: {exc}")
    return tables


def format_table_payload(table_name: str, table_df: pd.DataFrame, max_rows_per_table: int) -> Dict[str, object]:
    rows_to_take = max(1, min(max_rows_per_table, len(table_df))) if len(table_df) else 0
    sample_rows = table_df.head(rows_to_take).to_dict("records") if rows_to_take else []
    return {
        "table_name": table_name,
        "columns": list(table_df.columns),
        "sample_rows": sample_rows,
        "total_rows": int(len(table_df)),
    }


def build_prompt(table_id: str, table_payload: Dict[str, object]) -> str:
    table_payload_json = json.dumps(table_payload, indent=2, default=str)
    return f"""
Given the following table, generate 1 meaningful question-answer pair.
IMPORTANT: The question MUST be answerable using information from ONLY this table.
Only generate questions that require table information. Focus on pharmaceutical and medical aspects.

Table:
{table_payload_json}

Generate in the following format:
1. question: [specific question about the table data]
   short_answer: [one short direct answer, <= 25 words, explicitly answering the question]
   answer_reasoning: [brief justification using table evidence, 2-4 sentences]
   table_id: [{table_id}]
   table: [{table_id}]
"""


def generate_questions_for_table(
    *,
    client,
    table_id: str,
    table_df: pd.DataFrame,
    max_rows_per_table: int,
    model: str,
) -> List[Dict[str, str]]:
    payload = format_table_payload(table_id, table_df, max_rows_per_table=max_rows_per_table)

    prompt = build_prompt(table_id, payload)
    response = create_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical and pharmaceutical expert generating "
                    "table-grounded question-answer pairs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    table_id_str = table_id
    table_context_str = json.dumps(payload, ensure_ascii=False, default=str)

    qa_pairs = parse_llm_response(
        extract_response_text(response),
        pubmed_id=table_id_str,
        pubmed_text=table_context_str,
    )

    for row in qa_pairs:
        row["table_id"] = table_id_str
        row["table_content"] = table_context_str
        # Keep compatibility fields for shared helpers.
        row["pubmed_id"] = table_id_str
        row["pubmed_text"] = table_context_str
    return qa_pairs


def make_random_table_attempts(
    *,
    table_names: List[str],
    target_count: int,
    seed: Optional[int],
    limit: Optional[int],
) -> List[str]:
    if target_count <= 0:
        return []

    candidates = table_names[:]
    if limit is not None:
        candidates = candidates[: max(0, limit)]

    if not candidates:
        return []

    rng = random.Random(seed)
    if target_count <= len(candidates):
        return rng.sample(candidates, target_count)
    return [rng.choice(candidates) for _ in range(target_count)]


def run_sequential(
    *,
    model: str,
    drugbank_tables_dir: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    workers: int,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    max_rows_per_table: int,
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    del workers  # unused in sequential mode
    client = get_openrouter_client()

    tables = load_drugbank_tables(drugbank_tables_dir)
    table_names = list(tables.keys())

    attempts = make_random_table_attempts(
        table_names=table_names,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
    )

    id_generator = QuestionIdGenerator(prefix="tq")
    generated_writer = RollingCsvWriter(generated_file, TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)

    all_generated: List[Dict[str, str]] = []

    generation_pbar = tqdm(total=len(attempts), desc="Generation")
    for table_id in attempts:
        try:
            qa_pairs = generate_questions_for_table(
                client=client,
                table_id=table_id,
                table_df=tables[table_id],
                max_rows_per_table=max_rows_per_table,
                model=model,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for table {table_id}: {exc}")
            qa_pairs = []

        if qa_pairs:
            row = qa_pairs[0]
            row["question_id"] = id_generator.next_id()
            generated_writer.write(to_table_row(row))
            all_generated.append(row)

        generation_pbar.update(1)
    generation_pbar.close()

    processing_total = len(all_generated)
    processing_pbar = tqdm(total=processing_total, desc="Processing")
    correctness_pbar = tqdm(total=processing_total, desc="Correctness") if not skip_correctness else None
    paraphrase_pbar = tqdm(total=0, desc="Paraphrasing") if not skip_paraphrasing else None

    for row in all_generated:
        keep = True
        candidate_row = dict(row)

        if not skip_correctness:
            question = candidate_row.get("question", "")
            reference_answer = compose_reference_answer(candidate_row)
            llm_answer = generate_llm_answer(client, question, model=model)
            score = calculate_llm_correctness(
                client,
                llm_answer or "",
                reference_answer,
                question,
                model=model,
            )
            keep = score <= threshold
            if correctness_pbar is not None:
                correctness_pbar.update(1)

        if keep and not skip_paraphrasing:
            if paraphrase_pbar is not None:
                paraphrase_pbar.total += 1
                paraphrase_pbar.refresh()
            try:
                new_q = paraphrase_question_item(
                    client=client,
                    question=candidate_row.get("question", ""),
                    model=model,
                )
            except Exception as exc:  # pragma: no cover
                print(f"Error paraphrasing question: {exc}")
                new_q = None

            if paraphrase_pbar is not None:
                paraphrase_pbar.update(1)

            if not new_q:
                keep = False
            else:
                candidate_row["question"] = new_q

        if keep:
            processed_writer.write(to_table_row(candidate_row))

        processing_pbar.update(1)

    if correctness_pbar is not None:
        correctness_pbar.close()
    if paraphrase_pbar is not None:
        paraphrase_pbar.close()
    processing_pbar.close()

    generated_writer.close()
    processed_writer.close()

    print(f"Generated rows written: {generated_writer.count} -> {generated_file}")
    print(f"Processed rows written: {processed_writer.count} -> {processed_file}")


def run_parallel(
    *,
    model: str,
    drugbank_tables_dir: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    workers: int,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    max_rows_per_table: int,
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    client = get_openrouter_client()

    tables = load_drugbank_tables(drugbank_tables_dir)
    table_names = list(tables.keys())

    attempts = make_random_table_attempts(
        table_names=table_names,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
    )

    id_generator = QuestionIdGenerator(prefix="tq")

    generated_writer = RollingCsvWriter(generated_file, TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)

    queue: Queue[Optional[Dict[str, str]]] = Queue(maxsize=1024)

    progress_lock = threading.Lock()
    generation_pbar = tqdm(total=len(attempts), desc="Generation", position=0)
    processing_pbar = tqdm(total=0, desc="Processing", position=1)
    correctness_pbar = tqdm(total=0, desc="Correctness", position=2) if not skip_correctness else None
    paraphrase_pbar = tqdm(total=0, desc="Paraphrasing", position=3) if not skip_paraphrasing else None

    def producer_task(table_id: str) -> None:
        row: Optional[Dict[str, str]] = None
        try:
            qa_pairs = generate_questions_for_table(
                client=client,
                table_id=table_id,
                table_df=tables[table_id],
                max_rows_per_table=max_rows_per_table,
                model=model,
            )
            if qa_pairs:
                row = qa_pairs[0]
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for table {table_id}: {exc}")

        if row is not None:
            row["question_id"] = id_generator.next_id()
            generated_writer.write(to_table_row(row))
            queue.put(row)
            with progress_lock:
                processing_pbar.total += 1
                processing_pbar.refresh()
                if correctness_pbar is not None:
                    correctness_pbar.total += 1
                    correctness_pbar.refresh()
                elif paraphrase_pbar is not None:
                    paraphrase_pbar.total += 1
                    paraphrase_pbar.refresh()

        with progress_lock:
            generation_pbar.update(1)

    def consumer_task() -> None:
        while True:
            row = queue.get()
            try:
                if row is None:
                    return

                keep = True
                candidate_row = dict(row)

                if not skip_correctness:
                    question = candidate_row.get("question", "")
                    reference_answer = compose_reference_answer(candidate_row)
                    llm_answer = generate_llm_answer(client, question, model=model)
                    score = calculate_llm_correctness(
                        client,
                        llm_answer or "",
                        reference_answer,
                        question,
                        model=model,
                    )
                    keep = score <= threshold
                    if correctness_pbar is not None:
                        with progress_lock:
                            correctness_pbar.update(1)

                if keep and not skip_paraphrasing:
                    if paraphrase_pbar is not None and not skip_correctness:
                        with progress_lock:
                            paraphrase_pbar.total += 1
                            paraphrase_pbar.refresh()

                    try:
                        new_q = paraphrase_question_item(
                            client=client,
                            question=candidate_row.get("question", ""),
                            model=model,
                        )
                    except Exception as exc:  # pragma: no cover
                        print(f"Error paraphrasing question: {exc}")
                        new_q = None

                    if paraphrase_pbar is not None:
                        with progress_lock:
                            paraphrase_pbar.update(1)

                    if not new_q:
                        keep = False
                    else:
                        candidate_row["question"] = new_q

                if keep:
                    processed_writer.write(to_table_row(candidate_row))

                with progress_lock:
                    processing_pbar.update(1)
            finally:
                queue.task_done()

    consumers: List[threading.Thread] = []
    for _ in range(max(1, workers)):
        thread = threading.Thread(target=consumer_task, daemon=True)
        thread.start()
        consumers.append(thread)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(producer_task, attempts))

    for _ in consumers:
        queue.put(None)

    queue.join()
    for thread in consumers:
        thread.join()

    generation_pbar.close()
    processing_pbar.close()
    if correctness_pbar is not None:
        correctness_pbar.close()
    if paraphrase_pbar is not None:
        paraphrase_pbar.close()

    generated_writer.close()
    processed_writer.close()

    print(f"Generated rows written: {generated_writer.count} -> {generated_file}")
    print(f"Processed rows written: {processed_writer.count} -> {processed_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end table QA pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id for OpenRouter requests.")
    parser.add_argument("--drugbank-tables-dir", type=Path, default=DEFAULT_DRUGBANK_TABLES_DIR)
    parser.add_argument(
        "--generated-file",
        type=Path,
        default=DEFAULT_GENERATED,
        help="CSV output path for all generated QA rows.",
    )
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=DEFAULT_PROCESSED,
        help="CSV output path for processed QA rows (after correctness/paraphrasing).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Total number of generation attempts across random tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of tables considered when sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible table sampling.",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=20,
        help="Maximum rows to include per table in prompt context.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-paraphrasing", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (generation + processing). Use 1 for sequential.",
    )

    # Backward compatibility aliases.
    parser.add_argument("--filtered-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--paraphrased-file", type=Path, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def resolve_processed_file(args: argparse.Namespace) -> Path:
    if args.paraphrased_file is not None:
        return args.paraphrased_file
    if args.filtered_file is not None and args.skip_paraphrasing:
        return args.filtered_file
    return args.processed_file


def main() -> None:
    args = parse_args()
    processed_file = resolve_processed_file(args)

    if args.workers and args.workers > 1:
        run_parallel(
            model=args.model,
            drugbank_tables_dir=args.drugbank_tables_dir,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            workers=args.workers,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            max_rows_per_table=args.max_rows_per_table,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )
    else:
        run_sequential(
            model=args.model,
            drugbank_tables_dir=args.drugbank_tables_dir,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            workers=args.workers,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            max_rows_per_table=args.max_rows_per_table,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )


if __name__ == "__main__":
    main()
