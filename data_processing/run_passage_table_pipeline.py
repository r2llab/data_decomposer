"""End-to-end passage+table QA pipeline with optional parallel processing.

This script performs:
1) Question generation over a passage and its mapped table context.
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
from typing import Dict, List, Optional, Tuple

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
        load_drugbank_tables,
        load_pubmed_table_mapping,
        load_target_passages,
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
        load_drugbank_tables,
        load_pubmed_table_mapping,
        load_target_passages,
        paraphrase_question_item,
        parse_llm_response,
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PUBMED_TARGETS_DIR = BASE_DIR.parent / "data" / "Pharma" / "pubmed-targets"
DEFAULT_DRUGBANK_TABLES_DIR = BASE_DIR.parent / "data" / "Pharma" / "drugbank-tables"
DEFAULT_MAPPING_FILE = BASE_DIR.parent / "data" / "Pharma" / "pubmed-drugbank-tables.gt"

DEFAULT_GENERATED = BASE_DIR / "passage_table_generated.csv"
DEFAULT_PROCESSED = BASE_DIR / "passage_table_processed.csv"

PASSAGE_TABLE_FIELDNAMES = [
    "question_id",
    "question",
    "short_answer",
    "answer_reasoning",
    "pubmed_text",
    "pubmed_id",
    "table_content",
    "table_id",
]


def to_output_row(row: Dict[str, str]) -> Dict[str, str]:
    """Map parsed row fields to the passage+table output schema."""
    return {
        "question_id": row.get("question_id", ""),
        "question": row.get("question", ""),
        "short_answer": row.get("short_answer", ""),
        "answer_reasoning": row.get("answer_reasoning", ""),
        "pubmed_text": row.get("pubmed_text", ""),
        "pubmed_id": row.get("pubmed_id", ""),
        "table_content": row.get("table_content", ""),
        "table_id": row.get("table_id", ""),
    }


def normalize_table_name(table_name: str) -> str:
    return table_name.replace(".csv", "").strip()


def get_passage_table_names(
    *,
    passage_id: str,
    mapping: Dict[str, List[str]],
    tables,
    max_tables_per_passage: int,
) -> List[str]:
    raw_names = mapping.get(passage_id, [])
    names: List[str] = []
    seen = set()
    for name in raw_names:
        normalized = normalize_table_name(name)
        if not normalized or normalized in seen:
            continue
        if normalized not in tables:
            continue
        seen.add(normalized)
        names.append(normalized)

    if max_tables_per_passage > 0:
        names = names[:max_tables_per_passage]
    return names


def format_table_payload(table_name: str, table_df, max_rows_per_table: int) -> Dict[str, object]:
    rows_to_take = max(1, min(max_rows_per_table, len(table_df))) if len(table_df) else 0
    sample_rows = table_df.head(rows_to_take).to_dict("records") if rows_to_take else []
    return {
        "table_name": table_name,
        "columns": list(table_df.columns),
        "sample_rows": sample_rows,
        "total_rows": int(len(table_df)),
    }


def build_prompt(
    *,
    passage_id: str,
    passage_text: str,
    table_payloads: List[Dict[str, object]],
) -> str:
    clipped_passage = passage_text if len(passage_text) <= 1200 else passage_text[:1200] + "..."
    table_payloads_json = json.dumps(table_payloads, indent=2, default=str)
    table_ids = [payload.get("table_name", "") for payload in table_payloads]

    return f"""
Given the following passage and related tables, generate 1 meaningful question-answer pair.
IMPORTANT: Each question MUST require information from BOTH the passage and at least one table.
Only generate questions that need passage+table synthesis.
Focus on pharmaceutical and medical aspects.

Passage (ID: {passage_id}):
{clipped_passage}

Related tables:
{table_payloads_json}

Generate in the following format:
1. question: [specific question requiring passage + table information]
   short_answer: [one short direct answer, <= 25 words, explicitly answering the question]
   answer_reasoning: [brief justification using both passage and table evidence, 2-4 sentences]
   pubmed_id: [{passage_id}]
   table_id: [{', '.join(table_ids)}]
"""


def generate_questions_for_passage_table(
    *,
    client,
    passage_id: str,
    passage_text: str,
    table_ids: List[str],
    tables,
    max_rows_per_table: int,
    model: str,
) -> List[Dict[str, str]]:
    table_payloads = [
        format_table_payload(table_id, tables[table_id], max_rows_per_table=max_rows_per_table)
        for table_id in table_ids
    ]

    prompt = build_prompt(
        passage_id=passage_id,
        passage_text=passage_text,
        table_payloads=table_payloads,
    )

    response = create_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical and pharmaceutical expert generating "
                    "question-answer pairs that require combining passage and table evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    table_id_str = json.dumps(table_ids, ensure_ascii=False)
    table_content_str = json.dumps(table_payloads, ensure_ascii=False, default=str)

    qa_pairs = parse_llm_response(
        extract_response_text(response),
        pubmed_id=passage_id,
        pubmed_text=passage_text,
    )

    for row in qa_pairs:
        row["pubmed_id"] = passage_id
        row["pubmed_text"] = passage_text
        row["table_id"] = table_id_str
        row["table_content"] = table_content_str
    return qa_pairs


def make_random_passage_table_attempts(
    *,
    passages: Dict[str, str],
    mapping: Dict[str, List[str]],
    tables,
    target_count: int,
    seed: Optional[int],
    limit: Optional[int],
    max_tables_per_passage: int,
) -> List[Tuple[str, str, List[str]]]:
    if target_count <= 0:
        return []

    eligible: List[Tuple[str, str, List[str]]] = []
    for passage_id, passage_text in passages.items():
        table_names = get_passage_table_names(
            passage_id=passage_id,
            mapping=mapping,
            tables=tables,
            max_tables_per_passage=max_tables_per_passage,
        )
        if table_names:
            eligible.append((passage_id, passage_text, table_names))

    if limit is not None:
        eligible = eligible[: max(0, limit)]

    if not eligible:
        return []

    rng = random.Random(seed)
    if target_count <= len(eligible):
        return rng.sample(eligible, target_count)
    return [rng.choice(eligible) for _ in range(target_count)]


def run_sequential(
    *,
    model: str,
    pubmed_targets_dir: Path,
    drugbank_tables_dir: Path,
    mapping_file: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    max_tables_per_passage: int,
    max_rows_per_table: int,
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    tables = load_drugbank_tables(drugbank_tables_dir)
    mapping = load_pubmed_table_mapping(mapping_file)

    attempts = make_random_passage_table_attempts(
        passages=passages,
        mapping=mapping,
        tables=tables,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
        max_tables_per_passage=max_tables_per_passage,
    )

    id_generator = QuestionIdGenerator(prefix="ptq")
    generated_writer = RollingCsvWriter(generated_file, PASSAGE_TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, PASSAGE_TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)

    all_generated: List[Dict[str, str]] = []

    generation_pbar = tqdm(total=len(attempts), desc="Generation")
    for passage_id, passage_text, table_ids in attempts:
        try:
            qa_pairs = generate_questions_for_passage_table(
                client=client,
                passage_id=passage_id,
                passage_text=passage_text,
                table_ids=table_ids,
                tables=tables,
                max_rows_per_table=max_rows_per_table,
                model=model,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for {passage_id} with tables {table_ids}: {exc}")
            qa_pairs = []

        if qa_pairs:
            row = qa_pairs[0]
            row["question_id"] = id_generator.next_id()
            generated_writer.write(to_output_row(row))
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
            processed_writer.write(to_output_row(candidate_row))

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
    pubmed_targets_dir: Path,
    drugbank_tables_dir: Path,
    mapping_file: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    workers: int,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    max_tables_per_passage: int,
    max_rows_per_table: int,
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    tables = load_drugbank_tables(drugbank_tables_dir)
    mapping = load_pubmed_table_mapping(mapping_file)

    attempts = make_random_passage_table_attempts(
        passages=passages,
        mapping=mapping,
        tables=tables,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
        max_tables_per_passage=max_tables_per_passage,
    )

    id_generator = QuestionIdGenerator(prefix="ptq")

    generated_writer = RollingCsvWriter(generated_file, PASSAGE_TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, PASSAGE_TABLE_FIELDNAMES, flush_every=FLUSH_EVERY)

    queue: Queue[Optional[Dict[str, str]]] = Queue(maxsize=1024)

    progress_lock = threading.Lock()
    generation_pbar = tqdm(total=len(attempts), desc="Generation", position=0)
    processing_pbar = tqdm(total=0, desc="Processing", position=1)
    correctness_pbar = tqdm(total=0, desc="Correctness", position=2) if not skip_correctness else None
    paraphrase_pbar = tqdm(total=0, desc="Paraphrasing", position=3) if not skip_paraphrasing else None

    def producer_task(item: Tuple[str, str, List[str]]) -> None:
        passage_id, passage_text, table_ids = item

        row: Optional[Dict[str, str]] = None
        try:
            qa_pairs = generate_questions_for_passage_table(
                client=client,
                passage_id=passage_id,
                passage_text=passage_text,
                table_ids=table_ids,
                tables=tables,
                max_rows_per_table=max_rows_per_table,
                model=model,
            )
            if qa_pairs:
                row = qa_pairs[0]
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for {passage_id} with tables {table_ids}: {exc}")

        if row is not None:
            row["question_id"] = id_generator.next_id()
            generated_writer.write(to_output_row(row))
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
                    processed_writer.write(to_output_row(candidate_row))

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
    parser = argparse.ArgumentParser(description="Run the end-to-end passage+table QA pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id for OpenRouter requests.")
    parser.add_argument("--pubmed-targets-dir", type=Path, default=DEFAULT_PUBMED_TARGETS_DIR)
    parser.add_argument("--drugbank-tables-dir", type=Path, default=DEFAULT_DRUGBANK_TABLES_DIR)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
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
        help="Total number of generation attempts across random mapped passage-table contexts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of eligible passages considered for sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max-tables-per-passage",
        type=int,
        default=3,
        help="Maximum number of mapped tables to include per passage.",
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
            pubmed_targets_dir=args.pubmed_targets_dir,
            drugbank_tables_dir=args.drugbank_tables_dir,
            mapping_file=args.mapping_file,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            workers=args.workers,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            max_tables_per_passage=args.max_tables_per_passage,
            max_rows_per_table=args.max_rows_per_table,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )
    else:
        run_sequential(
            model=args.model,
            pubmed_targets_dir=args.pubmed_targets_dir,
            drugbank_tables_dir=args.drugbank_tables_dir,
            mapping_file=args.mapping_file,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            max_tables_per_passage=args.max_tables_per_passage,
            max_rows_per_table=args.max_rows_per_table,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )


if __name__ == "__main__":
    main()
