"""Verification and paraphrasing pipeline for passage-based QA datasets."""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = BASE_DIR / "passage_output.gt"
DEFAULT_FILTERED_OUTPUT = BASE_DIR / "passage_output_filtered.gt"
DEFAULT_PARAPHRASED_OUTPUT = BASE_DIR / "passage_output_paraphrased.gt"


def load_ground_truth(gt_file: Path) -> List[Dict[str, str]]:
    """Load ground truth data from a .gt CSV file."""
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_file}")

    with gt_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def generate_llm_answer(client: OpenAI, question: str, model: str = "gpt-4o") -> str:
    """Generate an answer for the provided question using an LLM."""
    try:
        prompt = f"Answer this question with detailed information: {question}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network interactions
        print(f"Error generating answer: {exc}")
        time.sleep(2)
        return ""


def calculate_llm_correctness(
    client: OpenAI,
    hypothesis: str,
    reference: str,
    question: str,
    model: str = "gpt-4o",
) -> float:
    """Ask the LLM to evaluate the correctness of a generated answer."""
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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        response_text = response.choices[0].message.content.strip()
        matches = re.findall(r"(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)(?:$|\s)", response_text)
        if matches:
            return float(matches[-1])
        print(f"Could not extract a score from LLM response: {response_text}")
        return 0.0
    except Exception as exc:  # pragma: no cover - network interactions
        print(f"Error in LLM evaluation: {exc}")
        time.sleep(2)
        return 0.0


def filter_qa_pairs(
    client: OpenAI,
    qa_data: Iterable[Dict[str, str]],
    llm_model: str = "gpt-4o",
    threshold: float = 0.5,
) -> List[Dict[str, str]]:
    """Filter QA pairs that the LLM can answer correctly above a threshold."""
    filtered: List[Dict[str, str]] = []
    for qa_pair in tqdm(list(qa_data), desc="Filtering QA pairs"):
        question = qa_pair["question"]
        reference_answer = qa_pair["answer"]

        llm_answer = generate_llm_answer(client, question, model=llm_model)
        if not llm_answer:
            filtered.append(qa_pair)
            continue

        correctness_score = calculate_llm_correctness(
            client, llm_answer, reference_answer, question, model=llm_model
        )
        print(f"Question: {question}")
        print(f"LLM Answer: {llm_answer[:100]}...")
        print(f"Correctness Score: {correctness_score}")

        if correctness_score <= threshold:
            filtered.append(qa_pair)
    return filtered


def save_gt_rows(rows: Iterable[Dict[str, str]], output_file: Path) -> None:
    """Persist QA rows to a CSV file in the expected .gt format."""
    rows = list(rows)
    if not rows:
        print("No data to save!")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def paraphrase_questions(
    client: OpenAI,
    input_csv: Path,
    output_csv: Path,
    model: str = "gpt-4o",
) -> List[Dict[str, str]]:
    """Paraphrase questions to remove references to specific tables or passages."""
    df = pd.read_csv(input_csv, quoting=csv.QUOTE_ALL)
    paraphrased_data: List[Dict[str, str]] = []

    for _, row in tqdm(list(df.iterrows()), desc="Paraphrasing questions"):
        row_dict = row.to_dict()
        question = row_dict.get("question", "")
        prompt = f"""
        Your task is to paraphrase the following question to remove references to specific tables, datasets, or passages.
        If the question is solely about dataset structure or tables (e.g., "Does the dataset provide any links...") return "REMOVE".
        Otherwise, paraphrase the question to make it more general while preserving its core content.

        Original question: {question}
        Paraphrased question:
        """
        response = client.chat.completions.create(
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
        paraphrased = response.choices[0].message.content.strip()
        if paraphrased == "REMOVE":
            continue
        row_dict["question"] = paraphrased
        paraphrased_data.append(row_dict)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = df.columns.tolist()
    if not paraphrased_data:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(fieldnames)
        print(
            "Processed %s questions, kept %s after paraphrasing and filtering"
            % (len(df), 0)
        )
        return paraphrased_data

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for item in paraphrased_data:
            writer.writerow({name: item.get(name, "") for name in fieldnames})

    print(
        "Processed %s questions, kept %s after paraphrasing and filtering"
        % (len(df), len(paraphrased_data))
    )
    return paraphrased_data


def run_pipeline(
    input_file: Path,
    filtered_output: Path,
    paraphrased_output: Path,
    model: str,
    threshold: float,
    skip_correctness: bool,
    skip_paraphrasing: bool,
) -> None:
    """Execute the correctness filtering and paraphrasing pipeline."""
    client = OpenAI()
    qa_data = load_ground_truth(input_file)
    print(f"Loaded {len(qa_data)} QA pairs from {input_file}")

    filtered_data = qa_data
    if not skip_correctness:
        filtered_data = filter_qa_pairs(client, qa_data, llm_model=model, threshold=threshold)
        print(f"Filtered to {len(filtered_data)} QA pairs")
        save_gt_rows(filtered_data, filtered_output)
        print(f"Saved filtered data to {filtered_output}")
    elif filtered_output != input_file:
        save_gt_rows(filtered_data, filtered_output)

    if skip_paraphrasing:
        return

    source_for_paraphrasing = filtered_output if filtered_output.exists() else input_file
    paraphrase_questions(client, source_for_paraphrasing, paraphrased_output, model=model)
    print(f"Saved paraphrased data to {paraphrased_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run correctness verification and paraphrasing on passage QA data.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input .gt file produced by question generation.",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=DEFAULT_FILTERED_OUTPUT,
        help="Destination for the correctness-filtered dataset.",
    )
    parser.add_argument(
        "--paraphrased-output",
        type=Path,
        default=DEFAULT_PARAPHRASED_OUTPUT,
        help="Destination for the paraphrased dataset.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model identifier to use for all LLM calls.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Correctness score threshold above which QA pairs are removed.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the correctness filtering step.",
    )
    parser.add_argument(
        "--skip-paraphrasing",
        action="store_true",
        help="Skip the paraphrasing step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_file=args.input_file,
        filtered_output=args.filtered_output,
        paraphrased_output=args.paraphrased_output,
        model=args.model,
        threshold=args.threshold,
        skip_correctness=args.skip_correctness,
        skip_paraphrasing=args.skip_paraphrasing,
    )


if __name__ == "__main__":
    main()
