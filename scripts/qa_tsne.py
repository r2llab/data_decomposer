#!/usr/bin/env python3
"""
qa_tsne.py
==========

Visualise a corpus of Q‑A pairs with OpenAI embeddings + t‑SNE.
Modified to process all .gt files from extracted_data directory and color by model.

Usage
-----
    python qa_tsne.py
    # choose which columns to embed:
    python qa_tsne.py --text "question+answer"
    # save 2‑D coords:
    python qa_tsne.py -o coords.csv

Requires
--------
pip install openai tiktoken pandas numpy scikit-learn matplotlib tqdm
"""

from __future__ import annotations
import argparse, os, sys, time, glob, re
from typing import List, Tuple

import pandas as pd
import numpy as np
from openai import OpenAI
import tiktoken
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────────────────
def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    filename_lower = filename.lower()
    if "gpt-4o-mini" in filename_lower:
        return "gpt-4o-mini"
    elif "gpt-4o" in filename_lower:
        return "gpt-4o"
    elif "qwen3-32b" in filename_lower or "qwen" in filename_lower:
        return "Qwen3-32B"
    else:
        return "unknown"


def load_all_gt_files(data_dir: str = "extracted_data") -> pd.DataFrame:
    """Load all .gt files from the data directory and combine them."""
    gt_files = glob.glob(os.path.join(data_dir, "*.gt"))
    
    if not gt_files:
        raise FileNotFoundError(f"No .gt files found in {data_dir}")
    
    all_data = []
    
    for file_path in gt_files:
        try:
            # First, try standard parsing
            df = pd.read_csv(file_path)
            model_name = extract_model_name(os.path.basename(file_path))
            df['model'] = model_name
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {os.path.basename(file_path)} (model: {model_name})")
        except Exception as e:
            print(f"Standard parsing failed for {file_path}: {e}")
            # Try custom parsing for malformed CSV
            try:
                df = load_malformed_csv(file_path)
                model_name = extract_model_name(os.path.basename(file_path))
                df['model'] = model_name
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
                print(f"Loaded {len(df)} rows from {os.path.basename(file_path)} (model: {model_name}) - used custom parsing")
            except Exception as e2:
                print(f"Failed to load {file_path} even with custom parsing: {e2}")
                continue
    
    if not all_data:
        raise ValueError("No valid .gt files could be loaded")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined data: {len(combined_df)} rows")
    print(f"Models found: {combined_df['model'].value_counts().to_dict()}")
    
    return combined_df


def load_malformed_csv(file_path: str) -> pd.DataFrame:
    """Load CSV files with malformed entries by fixing common issues."""
    import csv
    import io
    
    # Read the file and fix common issues
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Process each line to fix malformed CSV
    fixed_lines = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # Skip header
        if i == 0:
            fixed_lines.append(line)
            continue
            
        # Count quotes to detect malformed lines
        quote_count = line.count('"')
        
        # If odd number of quotes, likely malformed
        if quote_count % 2 != 0:
            # Try to fix by ensuring proper quote closure
            # This is a simple heuristic - add closing quote at end if needed
            if line.endswith(','):
                line = line[:-1] + '",None,"[\'unknown\']"'
            elif not line.endswith('"'):
                line = line + '"'
        
        # Ensure we have exactly 4 fields (question, answer, text, table)
        # Split by comma but respect quoted fields
        try:
            reader = csv.reader(io.StringIO(line))
            row = next(reader)
            
            # Pad or truncate to exactly 4 fields
            while len(row) < 4:
                row.append("None")
            if len(row) > 4:
                # Merge extra fields into the answer field
                row[1] = ' '.join(row[1:len(row)-2])
                row = row[:2] + row[-2:]
                
            # Reconstruct the line
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_ALL)
            writer.writerow(row)
            fixed_lines.append(output.getvalue().strip())
            
        except Exception:
            # If all else fails, skip this line
            print(f"Skipping malformed line {i+1} in {file_path}")
            continue
    
    # Create DataFrame from fixed content
    fixed_content = '\n'.join(fixed_lines)
    df = pd.read_csv(io.StringIO(fixed_content))
    
    return df


# ────────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ────────────────────────────────────────────────────────────────────────────────
def num_tokens(text: str, enc) -> int:
    """Count tokens in *text* for the target embedding model."""
    return len(enc.encode(text))


def embed_batch(
    client: OpenAI, batch: List[str], model_name: str
) -> List[List[float]]:
    """Call the embeddings endpoint once for an entire batch."""
    resp = client.embeddings.create(model=model_name, input=batch)
    # API returns embeddings in the same order
    return [item.embedding for item in resp.data]


def embed_corpus(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    max_tokens_per_batch: int = 8000,
) -> np.ndarray:
    """
    Embed *texts* with token‑aware batching so we never exceed the request limit.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from env
    enc = tiktoken.encoding_for_model(model_name)

    embeddings: List[List[float]] = []
    batch: List[str] = []
    tokens_in_batch = 0

    for text in tqdm(texts, desc="Embedding"):
        tks = num_tokens(text, enc)
        # Flush if this text won't fit
        if tokens_in_batch + tks > max_tokens_per_batch and batch:
            embeddings.extend(embed_batch(client, batch, model_name))
            batch, tokens_in_batch = [], 0

        batch.append(text)
        tokens_in_batch += tks

    # final flush
    if batch:
        embeddings.extend(embed_batch(client, batch, model_name))

    return np.asarray(embeddings, dtype="float32")


# ────────────────────────────────────────────────────────────────────────────────
# t‑SNE pipeline
# ────────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    df: pd.DataFrame,
    col_spec: str,
    model_name: str = "text-embedding-3-small",
    perplexity: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    • Build text per row according to *col_spec*  
      ("question", "answer", or "question+answer").  
    • Embed → PCA (50 D) → t‑SNE (2 D).  
    Returns (xy, texts, model_labels)
    """
    if col_spec not in {"question", "answer", "question+answer"}:
        raise ValueError("--text must be question, answer, or question+answer")

    if col_spec == "question":
        texts = df["question"].fillna("").tolist()
    elif col_spec == "answer":
        texts = df["answer"].fillna("").tolist()
    else:  # question+answer
        texts = (
            df["question"].fillna("") + " " + df["answer"].fillna("")
        ).tolist()

    model_labels = df["model"].values

    # 1. Embeddings
    emb = embed_corpus(texts, model_name=model_name)
    emb = normalize(emb)  # cosine distance -> Euclidean

    # 2. PCA to 50 D (speeds up + denoises)
    pca = PCA(n_components=50, random_state=0)
    emb50 = pca.fit_transform(emb)

    # 3. t‑SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=0,
    )
    xy = tsne.fit_transform(emb50)

    return xy, np.array(texts), model_labels


# ────────────────────────────────────────────────────────────────────────────────
# CLI + plotting
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="t‑SNE visualisation of Q‑A corpus from .gt files")
    p.add_argument(
        "--text",
        default="question",
        choices=["question", "answer", "question+answer"],
        help="Which text to embed per row (default: question)",
    )
    p.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )
    p.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t‑SNE perplexity (default: 30)",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional path to save a CSV containing x,y plus original row index and model",
    )
    p.add_argument(
        "--data-dir",
        default="extracted_data",
        help="Directory containing .gt files (default: extracted_data)",
    )
    p.add_argument(
        "--save-plot",
        default=None,
        help="Optional path to save the t-SNE plot as an image (e.g., tsne_plot.png)",
    )
    return p.parse_args()


def plot_xy_by_model(xy: np.ndarray, model_labels: np.ndarray, title: str = "", save_path: str = None) -> None:
    """Plot t-SNE results with different colors for each model."""
    plt.figure(figsize=(12, 10))
    
    # Define colors for each model
    model_colors = {
        "gpt-4o": "#1f77b4",      # blue
        "gpt-4o-mini": "#ff7f0e",  # orange
        "Qwen3-32B": "#2ca02c",    # green
        "unknown": "#d62728"       # red
    }
    
    unique_models = np.unique(model_labels)
    
    for model in unique_models:
        mask = model_labels == model
        color = model_colors.get(model, "#d62728")  # default to red for unknown
        plt.scatter(
            xy[mask, 0], 
            xy[mask, 1], 
            s=20, 
            alpha=0.7, 
            linewidths=0,
            color=color,
            label=f"{model} (n={np.sum(mask)})"
        )
    
    plt.title(title or "t‑SNE map colored by model")
    plt.xlabel("t‑SNE 1")
    plt.ylabel("t‑SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    
    plt.show()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set in environment.")

    # Load all .gt files
    df = load_all_gt_files(args.data_dir)
    
    if not {"question", "answer"}.issubset(df.columns):
        sys.exit("Error: CSV files must contain 'question' and 'answer' columns.")

    xy, texts, model_labels = run_pipeline(
        df, col_spec=args.text, model_name=args.model, perplexity=args.perplexity
    )

    # Plot
    title = f"t‑SNE of {args.text} ({args.model})"
    plot_xy_by_model(xy, model_labels, title, save_path=args.save_plot)

    # Optional CSV export
    if args.out:
        out_df = pd.DataFrame({
            "index": df.index,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "model": model_labels,
            "source_file": df["source_file"]
        })
        out_df.to_csv(args.out, index=False)
        print(f"Saved coordinates → {args.out}")


if __name__ == "__main__":
    main()
