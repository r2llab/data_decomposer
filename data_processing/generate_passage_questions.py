"""Utility for generating question-answer pairs from passage data using the OpenAI API."""
from __future__ import annotations

import argparse
import csv
import json
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


def load_passage_table_mapping(mapping_path: Path) -> Dict[str, List[str]]:
    """Load the mapping between passages and relevant tables."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    if not mapping_path.exists():
        return mapping

    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            passage_id, table_name = line.split(",", 1)
            mapping[passage_id].append(table_name)
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
    passage_id: str,
    passage_text: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Create the prompt used to request new question-answer pairs."""
    if len(passage_text) > 1000:
        passage_text = passage_text[:1000] + "..."

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

Passage (ID: {passage_id}):
{passage_text}

{context_str}

Generate questions in the following format:
1. question: [specific question about drug/treatment]
   answer: [detailed answer combining information from passage and tables]
   text: [passage ID if information from passage was used (e.g. Target-123456789)]
   table: [None]

Ensure every question uses information from the passage.
"""
    return prompt


def parse_llm_response(response_text: str, passage_id: str) -> List[Dict[str, str]]:
    """Parse an LLM response into structured question-answer dictionaries."""
    qa_pairs: List[Dict[str, str]] = []

    entries = response_text.strip().split("\n\n")
    for entry in entries:
        if not entry.strip():
            continue

        lines = entry.strip().split("\n")
        current = {"question": "", "answer": "", "text": passage_id, "table": "None"}

        for line in lines:
            line = line.strip()
            if not line or line.replace(".", "").strip().isdigit():
                continue

            if "question:" in line:
                current["question"] = line.split("question:", 1)[1].strip()
            elif "answer:" in line:
                current["answer"] = line.split("answer:", 1)[1].strip()
            elif "text:" in line:
                current["text"] = line.split("text:", 1)[1].strip()
            elif "table:" in line:
                table_value = line.split("table:", 1)[1].strip()
                if table_value.upper() in {"NA", "N/A", "NONE"}:
                    table_value = "None"
                current["table"] = table_value

        if current["question"] and current["answer"]:
            qa_pairs.append(current.copy())

    return qa_pairs


def generate_questions_for_passage(
    client: OpenAI,
    passage_id: str,
    passage_text: str,
    model: str,
    table_content: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    """Use the OpenAI client to generate question-answer pairs for a passage."""
    prompt = build_prompt(passage_id, passage_text, table_content)
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
    return parse_llm_response(content, passage_id)


def write_qa_pairs(output_file: Path, qa_pairs: Iterable[Dict[str, str]]) -> None:
    """Persist QA pairs to disk using the .gt CSV format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["question", "answer", "text", "table"])
        for qa_pair in qa_pairs:
            writer.writerow(
                [qa_pair["question"], qa_pair["answer"], qa_pair["text"], qa_pair["table"]]
            )


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
    mapping = load_passage_table_mapping(args.mapping_file)

    passage_items = list(passages.items())
    if args.limit is not None:
        passage_items = passage_items[: args.limit]

    all_pairs: List[Dict[str, str]] = []
    for passage_id, passage_text in tqdm(passage_items, desc="Generating questions"):
        relevant_tables = mapping.get(passage_id, [])
        table_content = None
        if relevant_tables and tables:
            table_content = get_relevant_table_content(
                tables, relevant_tables, max_rows=args.max_table_rows
            )
        qa_pairs = generate_questions_for_passage(
            client,
            passage_id,
            passage_text,
            model=args.model,
            table_content=table_content,
        )
        all_pairs.extend(qa_pairs)

    write_qa_pairs(args.output_file, all_pairs)
    print(f"Generated {len(all_pairs)} question-answer pairs -> {args.output_file}")


if __name__ == "__main__":
    main()
