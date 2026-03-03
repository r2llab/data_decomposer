"""End-to-end passage QA pipeline with optional parallel processing.

This script performs:
1) Question generation over passage content.
2) Optional correctness filtering.
3) Optional paraphrasing.

Outputs are written to CSV files (not .gt):
- generated file: all generated QA rows
- processed file: rows that pass optional processing stages
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
import time
from itertools import count
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    # Works when executed as a module: `python -m data_processing.run_passage_pipeline`.
    from .openrouter_client import create_chat_completion, get_openrouter_client
except ImportError:
    # Works when executed as a script: `python data_processing/run_passage_pipeline.py`.
    from openrouter_client import create_chat_completion, get_openrouter_client


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PUBMED_TARGETS_DIR = BASE_DIR.parent / "data" / "Pharma" / "pubmed-targets"
DEFAULT_DRUGBANK_TABLES_DIR = BASE_DIR.parent / "data" / "Pharma" / "drugbank-tables"
DEFAULT_MAPPING_FILE = BASE_DIR.parent / "data" / "Pharma" / "pubmed-drugbank-tables.gt"

DEFAULT_GENERATED = BASE_DIR / "passage_generated.csv"
DEFAULT_PROCESSED = BASE_DIR / "passage_processed.csv"
DEFAULT_MODEL = "openai/gpt-5.2"
FLUSH_EVERY = 20


GEN_FIELDNAMES = [
    "question_id",
    "question",
    "short_answer",
    "answer_reasoning",
    "pubmed_text",
    "pubmed_id",
]


class QuestionIdGenerator:
    """Thread-safe sequential ID generator for QA rows."""

    def __init__(self, prefix: str = "q") -> None:
        self._prefix = prefix
        self._counter = count(1)
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            value = next(self._counter)
        return f"{self._prefix}{value:08d}"


class RollingCsvWriter:
    """CSV writer that flushes to disk every N rows."""

    def __init__(self, path: Path, fieldnames: List[str], flush_every: int = FLUSH_EVERY) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.flush_every = max(1, flush_every)
        self._lock = threading.Lock()
        self._count = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=self.fieldnames,
            quoting=csv.QUOTE_ALL,
        )
        self._writer.writeheader()
        self._handle.flush()

    @property
    def count(self) -> int:
        return self._count

    def write(self, row: Dict[str, str]) -> None:
        with self._lock:
            self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
            self._count += 1
            if self._count % self.flush_every == 0:
                self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()


# ------------------------
# Data loading helpers
# ------------------------

def load_pubmed_table_mapping(mapping_path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not mapping_path.exists():
        return mapping
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pubmed_id, table_name = line.split(",", 1)
            mapping.setdefault(pubmed_id, []).append(table_name)
    return mapping


def load_target_passages(targets_dir: Path) -> Dict[str, str]:
    passages: Dict[str, str] = {}
    if not targets_dir.exists():
        raise FileNotFoundError(f"Passage directory does not exist: {targets_dir}")
    for file_path in sorted(targets_dir.glob("Target-*")):
        with file_path.open("r", encoding="utf-8") as handle:
            passages[file_path.name] = handle.read()
    return passages


def load_drugbank_tables(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if not tables_dir.exists():
        return tables
    for file_path in sorted(tables_dir.glob("*.csv")):
        tables[file_path.stem] = pd.read_csv(file_path)
    return tables


def get_relevant_table_content(
    tables: Dict[str, pd.DataFrame],
    table_names: Iterable[str],
    max_rows: int = 5,
) -> Dict[str, Dict[str, object]]:
    table_content: Dict[str, Dict[str, object]] = {}
    for table_name in table_names:
        base_table_name = table_name.replace(".csv", "")
        if base_table_name not in tables:
            continue
        df = tables[base_table_name]
        table_content[base_table_name] = {
            "columns": list(df.columns),
            "sample": df.head(max_rows).to_dict("records"),
        }
    return table_content


# ------------------------
# LLM helpers
# ------------------------

def build_prompt(
    pubmed_id: str,
    pubmed_text: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    if len(pubmed_text) > 1000:
        pubmed_text = pubmed_text[:1000] + "..."

    context_blocks: List[str] = []
    if table_content:
        context_blocks.append("Relevant table summaries:")
        context_blocks.append(json.dumps(table_content, indent=2))
    context_str = "\n".join(context_blocks)

    prompt = f"""
Given the following passage(s), generate 1 meaningful question-answer pair.
IMPORTANT: Each question MUST be answerable using information from ONLY the passage.
Only generate questions that require information from the passage.
Focus on pharmaceutical and medical aspects. Try and make the question as difficult and technical as possible.

PubMed passage (ID: {pubmed_id}):
{pubmed_text}

{context_str}

Generate questions in the following format:
1. question: [specific question about drug/treatment]
   short_answer: [one short direct answer, <= 25 words, explicitly answering the question]
   answer_reasoning: [brief justification using passage evidence, 2-4 sentences]
   pubmed_id: [passage ID if information from passage was used (e.g. Target-123456789)]
   table: [None]

Ensure every question uses information from the passage.
"""
    return prompt


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(part.strip() for part in parts if part).strip()
    return str(content).strip()


def extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    return _content_to_text(content)


def parse_llm_response(
    response_text: str,
    pubmed_id: str,
    pubmed_text: str,
) -> List[Dict[str, str]]:
    def new_record() -> Dict[str, str]:
        return {
            "question_id": "",
            "question": "",
            "short_answer": "",
            "answer_reasoning": "",
            "answer": "",
            "pubmed_text": pubmed_text,
            "passage_text": pubmed_text,
            "table": "None",
            "pubmed_id": pubmed_id,
        }

    if not response_text:
        return []

    qa_pairs: List[Dict[str, str]] = []
    current = new_record()
    active_field: Optional[str] = None

    label_pattern = re.compile(
        r"^\s*(?:-\s*)?(?:\d+\.\s*)?([A-Za-z_ ]+?)(?:\s*[:\-–]\s*)(.*)$"
    )

    def normalise_label(label: str) -> Optional[str]:
        label_key = label.lower().strip().replace(" ", "_")
        if label_key in {"q", "question"}:
            return "question"
        if label_key in {"short_answer", "direct_answer", "concise_answer", "final_answer"}:
            return "short_answer"
        if label_key in {"answer_reasoning", "reasoning", "rationale", "explanation"}:
            return "answer_reasoning"
        if label_key in {"a", "answer", "ans"}:
            return "answer"
        if label_key in {"pubmed_id", "pubmedid", "id"}:
            return "pubmed_id"
        if label_key in {"pubmed_text", "passage", "passage_text", "text"}:
            return "pubmed_text"
        if label_key in {"table", "table_id"}:
            return "table"
        return None

    def finalize_record(record: Dict[str, str]) -> Dict[str, str]:
        question = record.get("question", "").strip()
        short_answer = record.get("short_answer", "").strip()
        answer_reasoning = record.get("answer_reasoning", "").strip()
        answer = record.get("answer", "").strip()

        # Backward compatibility: if model only returns `answer`, keep a compact direct answer.
        if not short_answer and answer:
            sentence = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)[0]
            short_answer = sentence[:220].strip()

        if not answer_reasoning and answer:
            answer_reasoning = answer

        if not answer:
            if short_answer and answer_reasoning:
                answer = f"{short_answer} {answer_reasoning}".strip()
            else:
                answer = short_answer or answer_reasoning

        record["question"] = question
        record["short_answer"] = short_answer
        record["answer_reasoning"] = answer_reasoning
        record["answer"] = answer.strip()
        record["pubmed_text"] = record.get("pubmed_text", pubmed_text) or pubmed_text
        record["passage_text"] = record.get("passage_text", record["pubmed_text"]) or record["pubmed_text"]
        record["table"] = record.get("table", "None") or "None"
        record["pubmed_id"] = record.get("pubmed_id", pubmed_id) or pubmed_id
        return record

    def commit_current() -> None:
        nonlocal current, active_field
        current = finalize_record(current)
        has_answer = bool(
            current["short_answer"].strip()
            or current["answer_reasoning"].strip()
            or current["answer"].strip()
        )
        if current["question"].strip() and has_answer:
            qa_pairs.append(current)
        current = new_record()
        active_field = None

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = label_pattern.match(line)
        if match:
            label, value = match.groups()
            normalised = normalise_label(label)
            value = value.strip()

            if normalised == "question":
                has_partial_answer = bool(
                    current.get("short_answer")
                    or current.get("answer_reasoning")
                    or current.get("answer")
                )
                if current["question"] and has_partial_answer:
                    commit_current()
                current["question"] = value
                active_field = "question"
            elif normalised == "short_answer":
                current["short_answer"] = value
                active_field = "short_answer"
            elif normalised == "answer_reasoning":
                current["answer_reasoning"] = value
                active_field = "answer_reasoning"
            elif normalised == "answer":
                current["answer"] = value
                active_field = "answer"
            elif normalised == "pubmed_id":
                current["pubmed_id"] = value or pubmed_id
                active_field = None
            elif normalised == "pubmed_text":
                current["pubmed_text"] = value or pubmed_text
                current["passage_text"] = current["pubmed_text"]
                active_field = None
            elif normalised == "table":
                table_value = value or "None"
                if table_value.upper() in {"NA", "N/A", "NONE"}:
                    table_value = "None"
                current["table"] = table_value
                active_field = None
            else:
                active_field = None
            continue

        if active_field in {"question", "short_answer", "answer_reasoning", "answer"}:
            if current[active_field]:
                current[active_field] += f" {line}"
            else:
                current[active_field] = line

    current = finalize_record(current)
    has_answer = bool(
        current["short_answer"].strip()
        or current["answer_reasoning"].strip()
        or current["answer"].strip()
    )
    if current["question"].strip() and has_answer:
        qa_pairs.append(current)

    return qa_pairs


def compose_reference_answer(row: Dict[str, str]) -> str:
    """Compose reference answer used by correctness scoring."""
    short_answer = (row.get("short_answer") or "").strip()
    answer_reasoning = (row.get("answer_reasoning") or "").strip()
    answer = (row.get("answer") or "").strip()

    if short_answer and answer_reasoning:
        return f"{short_answer}\nReasoning: {answer_reasoning}"
    if answer:
        return answer
    return short_answer or answer_reasoning


def generate_questions_for_passage(
    client: Any,
    pubmed_id: str,
    pubmed_text: str,
    model: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    prompt = build_prompt(pubmed_id, pubmed_text, table_content)
    response = create_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical and pharmaceutical expert tasked with generating "
                    "detailed question-answer pairs about drugs and treatments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return parse_llm_response(extract_response_text(response), pubmed_id, pubmed_text)


def generate_llm_answer(client: Any, question: str, model: str = DEFAULT_MODEL) -> str:
    """Generate an answer for the provided question using an LLM."""
    try:
        prompt = f"Answer this question with detailed information: {question}"
        response = create_chat_completion(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return extract_response_text(response)
    except Exception as exc:  # pragma: no cover - network interactions
        print(f"Error generating answer: {exc}")
        time.sleep(1)
        return ""


def calculate_llm_correctness(
    client: Any,
    hypothesis: str,
    reference: str,
    question: str,
    model: str = DEFAULT_MODEL,
) -> float:
    """Ask the LLM to evaluate correctness of a generated answer."""
    try:
        prompt = f"""
You are an expert evaluator assessing the correctness of an answer to a question.

Question: {question}

Ground Truth Answer: {reference}

System Answer: {hypothesis}

Evaluate how correct the System Answer is compared to the Ground Truth Answer. Be very critical in your evaluation/analysis.
Give a score from 0 to 1 where:
- 1.0 means the System Answer is fully correct and contains all the information from the Ground Truth
- 0.0 means the System Answer is completely incorrect
- Values between 0 and 1 indicate partial correctness

Output a single line with just the score as a decimal between 0 and 1.
"""
        response = create_chat_completion(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        response_text = extract_response_text(response)
        matches = re.findall(r"(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)(?:$|\s)", response_text)
        if matches:
            return float(matches[-1])
        print(f"Could not extract a score from LLM response: {response_text}")
        return 0.0
    except Exception as exc:  # pragma: no cover - network interactions
        print(f"Error in LLM evaluation: {exc}")
        time.sleep(1)
        return 0.0


def paraphrase_question_item(*, client: Any, question: str, model: str) -> Optional[str]:
    """Paraphrase a single question (returns None to drop the item)."""
    prompt = f"""
Your task is to paraphrase the following question to remove references to specific tables, datasets, or passages.
If the question is solely about dataset structure or tables (e.g., "Does the dataset provide any links...") return "REMOVE".
Otherwise, paraphrase the question to make it more general while preserving its core content.

Original question: {question}
Paraphrased question:
"""
    response = create_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in reformulating pharmaceutical questions to make them more general "
                    "and independent of specific data sources."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    text = extract_response_text(response)
    if not text:
        return None
    if text.strip().upper() == "REMOVE":
        return None
    return text


def make_random_attempts(
    pubmed_items: List[Tuple[str, str]],
    target_count: int,
    seed: Optional[int],
) -> List[Tuple[str, str]]:
    """Randomly select passages from the available pool."""
    if target_count <= 0 or not pubmed_items:
        return []

    rng = random.Random(seed)
    if target_count <= len(pubmed_items):
        return rng.sample(pubmed_items, target_count)
    return [rng.choice(pubmed_items) for _ in range(target_count)]


# ------------------------
# Pipeline execution
# ------------------------

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
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    """Run generation and processing in a single thread."""
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    tables = load_drugbank_tables(drugbank_tables_dir)
    mapping = load_pubmed_table_mapping(mapping_file)

    pubmed_items = list(passages.items())
    if limit is not None:
        pubmed_items = pubmed_items[:limit]

    target_count = max(0, num_questions)
    id_generator = QuestionIdGenerator()
    generated_writer = RollingCsvWriter(generated_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)

    all_generated: List[Dict[str, str]] = []

    attempts = make_random_attempts(pubmed_items, target_count, seed)
    generation_pbar = tqdm(total=target_count, desc="Generation")
    for pubmed_id, pubmed_text in attempts:
        relevant_tables = mapping.get(pubmed_id, [])
        table_content = None
        if relevant_tables and tables:
            table_content = get_relevant_table_content(tables, relevant_tables)

        try:
            qa_pairs = generate_questions_for_passage(
                client,
                pubmed_id,
                pubmed_text,
                model=model,
                table_content=table_content,
            )
        except Exception as exc:  # pragma: no cover - network interactions
            print(f"Error generating questions for {pubmed_id}: {exc}")
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
    paraphrase_pbar: Optional[tqdm] = tqdm(total=0, desc="Paraphrasing") if not skip_paraphrasing else None

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
            except Exception as exc:  # pragma: no cover - network interactions
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
    drugbank_tables_dir: Path,
    mapping_file: Path,
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
    """Run generation and processing concurrently with thread workers."""
    client = get_openrouter_client()

    passages = load_target_passages(pubmed_targets_dir)
    tables = load_drugbank_tables(drugbank_tables_dir)
    mapping = load_pubmed_table_mapping(mapping_file)

    pubmed_items = list(passages.items())
    if limit is not None:
        pubmed_items = pubmed_items[:limit]

    target_count = max(0, num_questions)
    id_generator = QuestionIdGenerator()

    generated_writer = RollingCsvWriter(generated_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)
    processed_writer = RollingCsvWriter(processed_file, GEN_FIELDNAMES, flush_every=FLUSH_EVERY)

    queue: Queue[Optional[Dict[str, str]]] = Queue(maxsize=1024)

    progress_lock = threading.Lock()
    generation_pbar = tqdm(total=target_count, desc="Generation", position=0)
    processing_pbar = tqdm(total=0, desc="Processing", position=1)
    correctness_pbar: Optional[tqdm] = None
    paraphrase_pbar: Optional[tqdm] = None

    if not skip_correctness:
        correctness_pbar = tqdm(total=0, desc="Correctness", position=2)
    if not skip_paraphrasing:
        paraphrase_pbar = tqdm(total=0, desc="Paraphrasing", position=3)

    def producer_task(item: Tuple[str, str]) -> None:
        pubmed_id, pubmed_text = item

        relevant_tables = mapping.get(pubmed_id, [])
        table_content = None
        if relevant_tables and tables:
            table_content = get_relevant_table_content(tables, relevant_tables)

        row: Optional[Dict[str, str]] = None
        try:
            qa_pairs = generate_questions_for_passage(
                client,
                pubmed_id,
                pubmed_text,
                model=model,
                table_content=table_content,
            )
            if qa_pairs:
                row = qa_pairs[0]
        except Exception as exc:  # pragma: no cover - network interactions
            print(f"Error generating questions for {pubmed_id}: {exc}")

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
                    except Exception as exc:  # pragma: no cover - network interactions
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

    attempts = make_random_attempts(pubmed_items, target_count, seed)
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
    parser = argparse.ArgumentParser(description="Run the end-to-end passage QA pipeline.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id for OpenRouter requests.",
    )
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
        help="Total number of generation attempts across passages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max passages to consider as sources (debug).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible passage sampling.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-paraphrasing", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (generation + processing). Use 1 for sequential.",
    )

    # Backward compatibility aliases from older versions.
    parser.add_argument("--filtered-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--paraphrased-file", type=Path, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def resolve_processed_file(args: argparse.Namespace) -> Path:
    """Resolve output path while supporting older CLI flags."""
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
            skip_correctness=args.skip_correctness,
            skip_paraphrasing=args.skip_paraphrasing,
        )


if __name__ == "__main__":
    main()
