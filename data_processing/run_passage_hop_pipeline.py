"""End-to-end passage-hop QA pipeline with optional parallel processing.

This script performs:
1) Question generation over pairs of related passages.
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

try:
    from .openrouter_client import create_chat_completion, get_openrouter_client
    from .run_passage_pipeline import (
        DEFAULT_MODEL,
        FLUSH_EVERY,
        GEN_FIELDNAMES,
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
        GEN_FIELDNAMES,
        QuestionIdGenerator,
        RollingCsvWriter,
        calculate_llm_correctness,
        compose_reference_answer,
        extract_response_text,
        generate_llm_answer,
        paraphrase_question_item,
        parse_llm_response,
    )

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PUBMED_TARGETS_DIR = BASE_DIR.parent / "data" / "Pharma" / "pubmed-targets"
DEFAULT_GROUPED_PASSAGES_FILE = BASE_DIR.parent / "grouped_passages_by_drug.json"

DEFAULT_GENERATED = BASE_DIR / "passage_hop_generated.csv"
DEFAULT_PROCESSED = BASE_DIR / "passage_hop_processed.csv"


def load_target_passages(targets_dir: Path) -> Dict[str, str]:
    passages: Dict[str, str] = {}
    if not targets_dir.exists():
        raise FileNotFoundError(f"Passage directory does not exist: {targets_dir}")

    for file_path in sorted(targets_dir.glob("Target-*")):
        with file_path.open("r", encoding="utf-8") as handle:
            passages[file_path.name] = handle.read()
    return passages


def load_grouped_passages(grouped_file: Path) -> Dict[str, List[str]]:
    if not grouped_file.exists():
        raise FileNotFoundError(f"Grouped passage file does not exist: {grouped_file}")
    with grouped_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    normalized: Dict[str, List[str]] = {}
    for drug, ids in data.items():
        if not isinstance(drug, str) or not isinstance(ids, list):
            continue
        normalized[drug] = [str(pid) for pid in ids if isinstance(pid, str)]
    return normalized


def build_passage_bundle(passage_ids: List[str], passage_texts: List[str]) -> str:
    chunks: List[str] = []
    for idx, (pid, text) in enumerate(zip(passage_ids, passage_texts), 1):
        text_block = text if len(text) <= 1200 else text[:1200] + "..."
        chunks.append(f"Passage {idx} (ID: {pid}):\n{text_block}")
    return "\n\n".join(chunks)


def build_prompt(drug_name: str, passage_ids: List[str], passage_bundle: str) -> str:
    return f"""
Given the following passages about the same drug ({drug_name}), generate 1 meaningful multi-hop question-answer pair.
IMPORTANT: The question MUST require synthesizing information from BOTH passages to answer correctly.
Do not generate questions answerable from only one passage.
Focus on pharmaceutical and medical aspects. Keep the answer concise but technically grounded.

{passage_bundle}

Generate in the following format:
1. question: [specific question requiring both passages]
   short_answer: [one short direct answer, <= 25 words, explicitly answering the question]
   answer_reasoning: [brief justification using evidence from both passages, 2-4 sentences]
   pubmed_id: [{', '.join(passage_ids)}]
   table: [None]
"""


def generate_questions_for_passage_pair(
    *,
    client,
    drug_name: str,
    passage_ids: List[str],
    passage_texts: List[str],
    model: str,
) -> List[Dict[str, str]]:
    bundle = build_passage_bundle(passage_ids, passage_texts)
    prompt = build_prompt(drug_name, passage_ids, bundle)

    response = create_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical and pharmaceutical expert generating "
                    "high-quality multi-hop question-answer pairs from multiple passages."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    pair_ids_json = json.dumps(passage_ids, ensure_ascii=False)
    passage_texts_json = json.dumps(passage_texts, ensure_ascii=False)
    qa_pairs = parse_llm_response(
        extract_response_text(response),
        pubmed_id=pair_ids_json,
        pubmed_text=passage_texts_json,
    )

    for row in qa_pairs:
        row["pubmed_id"] = pair_ids_json
        row["pubmed_text"] = passage_texts_json
    return qa_pairs


def make_random_hop_attempts(
    *,
    passages: Dict[str, str],
    grouped_passages: Dict[str, List[str]],
    target_count: int,
    seed: Optional[int],
    limit: Optional[int],
) -> List[Tuple[str, List[str], List[str]]]:
    if target_count <= 0:
        return []

    eligible: List[Tuple[str, List[str]]] = []
    for drug, ids in grouped_passages.items():
        valid_ids = [pid for pid in ids if pid in passages]
        # Preserve order but de-duplicate.
        unique_valid_ids = list(dict.fromkeys(valid_ids))
        if len(unique_valid_ids) >= 2:
            eligible.append((drug, unique_valid_ids))

    if limit is not None:
        eligible = eligible[: max(0, limit)]

    if not eligible:
        return []

    rng = random.Random(seed)
    attempts: List[Tuple[str, List[str], List[str]]] = []
    for _ in range(target_count):
        drug, ids = rng.choice(eligible)
        pair_ids = rng.sample(ids, 2)
        pair_texts = [passages[pid] for pid in pair_ids]
        attempts.append((drug, pair_ids, pair_texts))
    return attempts


def run_sequential(
    *,
    model: str,
    pubmed_targets_dir: Path,
    grouped_passages_file: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    grouped_passages = load_grouped_passages(grouped_passages_file)

    attempts = make_random_hop_attempts(
        passages=passages,
        grouped_passages=grouped_passages,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
    )

    id_generator = QuestionIdGenerator(prefix="phq")
    generated_writer = RollingCsvWriter(generated_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)

    all_generated: List[Dict[str, str]] = []

    generation_pbar = tqdm(total=len(attempts), desc="Generation")
    for drug_name, pair_ids, pair_texts in attempts:
        try:
            qa_pairs = generate_questions_for_passage_pair(
                client=client,
                drug_name=drug_name,
                passage_ids=pair_ids,
                passage_texts=pair_texts,
                model=model,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for {drug_name} {pair_ids}: {exc}")
            qa_pairs = []

        if qa_pairs:
            row = qa_pairs[0]
            row["question_id"] = id_generator.next_id()
            generated_writer.write(row)
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
            processed_writer.write(candidate_row)

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
    grouped_passages_file: Path,
    generated_file: Path,
    processed_file: Path,
    threshold: float,
    workers: int,
    num_questions: int,
    limit: Optional[int],
    seed: Optional[int],
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    grouped_passages = load_grouped_passages(grouped_passages_file)

    attempts = make_random_hop_attempts(
        passages=passages,
        grouped_passages=grouped_passages,
        target_count=max(0, num_questions),
        seed=seed,
        limit=limit,
    )

    id_generator = QuestionIdGenerator(prefix="phq")

    generated_writer = RollingCsvWriter(generated_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)

    queue: Queue[Optional[Dict[str, str]]] = Queue(maxsize=1024)

    progress_lock = threading.Lock()
    generation_pbar = tqdm(total=len(attempts), desc="Generation", position=0)
    processing_pbar = tqdm(total=0, desc="Processing", position=1)
    correctness_pbar = tqdm(total=0, desc="Correctness", position=2) if not skip_correctness else None
    paraphrase_pbar = tqdm(total=0, desc="Paraphrasing", position=3) if not skip_paraphrasing else None

    def producer_task(item: Tuple[str, List[str], List[str]]) -> None:
        drug_name, pair_ids, pair_texts = item

        row: Optional[Dict[str, str]] = None
        try:
            qa_pairs = generate_questions_for_passage_pair(
                client=client,
                drug_name=drug_name,
                passage_ids=pair_ids,
                passage_texts=pair_texts,
                model=model,
            )
            if qa_pairs:
                row = qa_pairs[0]
        except Exception as exc:  # pragma: no cover
            print(f"Error generating questions for {drug_name} {pair_ids}: {exc}")

        if row is not None:
            row["question_id"] = id_generator.next_id()
            generated_writer.write(row)
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
                    processed_writer.write(candidate_row)

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
    parser = argparse.ArgumentParser(description="Run the end-to-end passage-hop QA pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id for OpenRouter requests.")
    parser.add_argument("--pubmed-targets-dir", type=Path, default=DEFAULT_PUBMED_TARGETS_DIR)
    parser.add_argument("--grouped-passages-file", type=Path, default=DEFAULT_GROUPED_PASSAGES_FILE)
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
        help="Total number of generation attempts across random passage pairs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of eligible drug groups considered.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible pair sampling.",
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
            grouped_passages_file=args.grouped_passages_file,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            workers=args.workers,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )
    else:
        run_sequential(
            model=args.model,
            pubmed_targets_dir=args.pubmed_targets_dir,
            grouped_passages_file=args.grouped_passages_file,
            generated_file=args.generated_file,
            processed_file=processed_file,
            threshold=args.threshold,
            num_questions=args.num_questions,
            limit=args.limit,
            seed=args.seed,
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )


if __name__ == "__main__":
    main()
