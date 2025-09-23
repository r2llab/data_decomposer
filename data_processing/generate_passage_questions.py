"""Utility for generating question-answer pairs from passage data using the OpenAI API."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PUBMED_TARGETS_DIR = BASE_DIR.parent / "data" / "Pharma" / "pubmed-targets"
DEFAULT_DRUGBANK_TABLES_DIR = BASE_DIR.parent / "data" / "Pharma" / "drugbank-tables"
DEFAULT_MAPPING_FILE = BASE_DIR.parent / "data" / "Pharma" / "pubmed-drugbank-tables.gt"
DEFAULT_OUTPUT_FILE = BASE_DIR / "passage_output.gt"


def load_pubmed_table_mapping(mapping_path: Path) -> Dict[str, List[str]]:
    """Load the mapping between PubMed passages and relevant tables."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    if not mapping_path.exists():
        return mapping

    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pubmed_id, table_name = line.split(",", 1)
            mapping[pubmed_id].append(table_name)
    return mapping


def load_target_passages(targets_dir: Path) -> Dict[str, str]:
    """Load all target passages from the configured directory."""
    passages: Dict[str, str] = {}
    if not targets_dir.exists():
        raise FileNotFoundError(f"Passage directory does not exist: {targets_dir}")

    for file_path in sorted(targets_dir.glob("Target-*")):
        with file_path.open("r", encoding="utf-8") as handle:
            passages[file_path.name] = handle.read()
    return passages


def load_drugbank_tables(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load DrugBank tables as DataFrames keyed by table stem name."""
    tables: Dict[str, pd.DataFrame] = {}
    if not tables_dir.exists():
        return tables

    for file_path in sorted(tables_dir.glob("*.csv")):
        tables[file_path.stem] = pd.read_csv(file_path)
    return tables


def get_relevant_table_content(
    tables: Dict[str, pd.DataFrame], table_names: Iterable[str], max_rows: int = 5
) -> Dict[str, Dict[str, object]]:
    """Extract a light-weight summary of table content for prompt construction."""
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


def build_prompt(
    pubmed_id: str,
    pubmed_text: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Create the prompt used to request new question-answer pairs."""
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
   answer: [detailed answer combining information from passage and tables]
   pubmed_id: [passage ID if information from passage was used (e.g. Target-123456789)]
   table: [None]

Ensure every question uses information from the passage.
"""
    return prompt


def parse_llm_response(
    response_text: str, pubmed_id: str, pubmed_text: str
) -> List[Dict[str, str]]:
    """Parse an LLM response into structured question-answer dictionaries."""

    def new_record() -> Dict[str, str]:
        return {
            "question": "",
            "answer": "",
            "pubmed_text": pubmed_text,
            "table": "None",
            "pubmed_id": pubmed_id,
        }

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
        if label_key in {"a", "answer", "ans"}:
            return "answer"
        if label_key in {"pubmed_id", "pubmedid", "id"}:
            return "pubmed_id"
        if label_key in {"pubmed_text", "passage", "passage_text", "text"}:
            return "pubmed_text"
        if label_key in {"table", "table_id"}:
            return "table"
        return None

    def commit_current() -> None:
        nonlocal current, active_field
        if current["question"].strip() and current["answer"].strip():
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
                if current["question"] and current["answer"]:
                    commit_current()
                current["question"] = value
                active_field = "question"
            elif normalised == "answer":
                current["answer"] = value
                active_field = "answer"
            elif normalised == "pubmed_id":
                current["pubmed_id"] = value or pubmed_id
                active_field = None
            elif normalised == "pubmed_text":
                current["pubmed_text"] = value or pubmed_text
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

        if active_field in {"question", "answer"}:
            if current[active_field]:
                current[active_field] += f" {line}"
            else:
                current[active_field] = line

    if current["question"].strip() and current["answer"].strip():
        qa_pairs.append(current)

    return qa_pairs


def generate_questions_for_passage(
    client: OpenAI,
    pubmed_id: str,
    pubmed_text: str,
    model: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    """Use the OpenAI client to generate question-answer pairs for a passage."""
    prompt = build_prompt(pubmed_id, pubmed_text, table_content)
    response = client.chat.completions.create(
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
    content = response.choices[0].message.content
    return parse_llm_response(content, pubmed_id, pubmed_text)


def write_qa_pairs(output_file: Path, qa_pairs: Iterable[Dict[str, str]]) -> None:
    """Persist QA pairs to disk using the .gt CSV format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    qa_pairs = list(qa_pairs)
    if not qa_pairs:
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(["question", "answer", "pubmed_text", "table", "pubmed_id"])
        return

    fieldnames = ["question", "answer", "pubmed_text", "table", "pubmed_id"]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for qa_pair in qa_pairs:
            writer.writerow({name: qa_pair.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate passage-based QA pairs.")
    parser.add_argument(
        "--pubmed-targets-dir",
        type=Path,
        default=DEFAULT_PUBMED_TARGETS_DIR,
        help="Directory containing passage text files.",
    )
    parser.add_argument(
        "--drugbank-tables-dir",
        type=Path,
        default=DEFAULT_DRUGBANK_TABLES_DIR,
        help="Directory containing DrugBank CSV tables.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help="Optional mapping file relating passages to tables.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination .gt file for generated QA pairs.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model identifier used for generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of passages to process.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=5,
        help="Number of rows to include when summarising tables.",
    )

    args = parser.parse_args()

    client = OpenAI()

    passages = load_target_passages(args.pubmed_targets_dir)
    tables = load_drugbank_tables(args.drugbank_tables_dir)
    mapping = load_pubmed_table_mapping(args.mapping_file)

    pubmed_items = list(passages.items())
    if args.limit is not None:
        pubmed_items = pubmed_items[: args.limit]

    all_pairs: List[Dict[str, str]] = []
    for pubmed_id, pubmed_text in tqdm(pubmed_items, desc="Generating questions"):
        relevant_tables = mapping.get(pubmed_id, [])
        table_content = None
        if relevant_tables and tables:
            table_content = get_relevant_table_content(
                tables, relevant_tables, max_rows=args.max_table_rows
            )
        qa_pairs = generate_questions_for_passage(
            client,
            pubmed_id,
            pubmed_text,
            model=args.model,
            table_content=table_content,
        )
        all_pairs.extend(qa_pairs)

    write_qa_pairs(args.output_file, all_pairs)
    print(f"Generated {len(all_pairs)} question-answer pairs -> {args.output_file}")


if __name__ == "__main__":
    main()
