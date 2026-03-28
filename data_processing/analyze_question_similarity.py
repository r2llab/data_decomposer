#!/usr/bin/env python3
"""
Analyze diversity/similarity of processed generated questions.

This script reads processed question CSV files (default: questions_final/*_processed.csv),
computes sentence embeddings for question text, and reports:
1) overall cosine-similarity distribution
2) per-question-type breakdown (counts + within-type similarity stats)
3) 2D t-SNE visualization colored by question type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .openrouter_client import get_openrouter_client
except ImportError:
    from openrouter_client import get_openrouter_client

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None


FILE_TYPE_HINTS: List[Tuple[str, str]] = [
    ("passage_table", "passage_table"),
    ("passage_hop", "passage_hop"),
    ("table_hop", "table_hop"),
    ("passage", "passage"),
    ("table", "table"),
]

ID_PREFIX_HINTS: List[Tuple[str, str]] = [
    ("ptq", "passage_table"),
    ("phq", "passage_hop"),
    ("thq", "table_hop"),
    ("tq", "table"),
    ("q", "passage"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cosine-similarity distribution over processed questions."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("questions_final"),
        help="Directory containing question CSV files.",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*_processed.csv",
        help="Glob pattern for processed files to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/question_similarity"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="question",
        help="Text column used for embeddings/similarity.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="question_id",
        help="Question ID column.",
    )
    parser.add_argument(
        "--question-type-column",
        type=str,
        default="question_type",
        help="Question type column if present. Falls back to file/ID inference.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="openai/text-embedding-3-large",
        help="OpenRouter embedding model name (OpenAI-compatible).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Optional cap on number of questions (0 = use all).",
    )
    parser.add_argument(
        "--max-exact-pairs",
        type=int,
        default=3_000_000,
        help="Max number of pairs to compute exactly before sampling.",
    )
    parser.add_argument(
        "--overall-sample-pairs",
        type=int,
        default=300_000,
        help="Sample size for overall pairwise similarities when not exact.",
    )
    parser.add_argument(
        "--per-type-sample-pairs",
        type=int,
        default=100_000,
        help="Sample size per question type when within-type pairs are not exact.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=2500,
        help="Max points shown in the t-SNE scatter plot.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (auto-adjusted for sample size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and t-SNE.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip creating plot files.",
    )
    return parser.parse_args()


def infer_type_from_file_name(file_name: str) -> str:
    lowered = file_name.lower()
    for needle, question_type in FILE_TYPE_HINTS:
        if needle in lowered:
            return question_type
    return "unknown"


def infer_type_from_question_id(question_id: str) -> str:
    lowered = question_id.strip().lower()
    for prefix, question_type in ID_PREFIX_HINTS:
        if lowered.startswith(prefix):
            return question_type
    return "unknown"


def build_question_type(row: pd.Series, type_column: str) -> str:
    from_column = str(row.get(type_column, "") or "").strip().lower()
    if from_column:
        return from_column

    from_id = infer_type_from_question_id(str(row.get("question_id", "") or ""))
    if from_id != "unknown":
        return from_id

    return infer_type_from_file_name(str(row.get("source_file", "")))


def load_processed_questions(
    input_dir: Path,
    file_glob: str,
    text_column: str,
    id_column: str,
    type_column: str,
) -> Tuple[pd.DataFrame, List[Path]]:
    files = sorted(input_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matched '{file_glob}' in '{input_dir.resolve()}'."
        )

    frames: List[pd.DataFrame] = []
    for file_path in tqdm(files, desc="Loading processed files"):
        frame = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        if text_column not in frame.columns:
            print(f"Skipping {file_path.name}: missing text column '{text_column}'")
            continue

        if id_column not in frame.columns:
            frame[id_column] = ""

        frame["source_file"] = file_path.name
        frame[text_column] = frame[text_column].fillna("").astype(str).str.strip()
        frame[id_column] = frame[id_column].fillna("").astype(str).str.strip()
        frame = frame[frame[text_column] != ""].copy()
        frame["question_type"] = frame.apply(
            lambda row: build_question_type(row, type_column), axis=1
        )
        frames.append(frame)

    if not frames:
        raise ValueError(
            "No valid processed rows were loaded. Check file format and text column."
        )

    combined = pd.concat(frames, ignore_index=True)

    if id_column in combined.columns:
        has_id = combined[id_column].astype(str).str.strip() != ""
        combined = pd.concat(
            [
                combined[has_id]
                .drop_duplicates(subset=[id_column], keep="first")
                .reset_index(drop=True),
                combined[~has_id].drop_duplicates(subset=[text_column], keep="first"),
            ],
            ignore_index=True,
        )
        combined = combined.drop_duplicates(subset=[text_column], keep="first")

    return combined.reset_index(drop=True), files


def encode_questions(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    client = get_openrouter_client()
    print(f"Encoding {len(texts)} questions via OpenRouter model: {model_name}")

    vectors: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding batches"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = client.embeddings.create(
                    model=model_name,
                    input=batch_texts,
                )
                vectors.extend(item.embedding for item in response.data)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                sleep_seconds = min(10.0, 1.5**attempt)
                print(
                    f"Embedding batch failed (attempt {attempt}/5). "
                    f"Retrying in {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)

        if last_error is not None:
            raise RuntimeError(
                f"Failed embedding batch after retries: {last_error}"
            ) from last_error

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Unexpected embeddings shape from OpenRouter: {embeddings.shape}"
        )

    # L2-normalize so dot product equals cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def summarize_similarities(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "pairs_used": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "fraction_ge_0_8": None,
            "fraction_ge_0_9": None,
        }

    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "pairs_used": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "max": float(np.max(values)),
        "fraction_ge_0_8": float(np.mean(values >= 0.8)),
        "fraction_ge_0_9": float(np.mean(values >= 0.9)),
    }


def sample_pairwise_similarities(
    embeddings: np.ndarray,
    sample_pairs: int,
    seed: int,
) -> np.ndarray:
    num_questions = embeddings.shape[0]
    if num_questions < 2 or sample_pairs <= 0:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    needed = sample_pairs
    idx_a_chunks: List[np.ndarray] = []
    idx_b_chunks: List[np.ndarray] = []

    while needed > 0:
        draw_size = max(needed * 2, 1_000)
        idx_a = rng.integers(0, num_questions, size=draw_size)
        idx_b = rng.integers(0, num_questions, size=draw_size)
        valid = idx_a != idx_b
        idx_a = idx_a[valid]
        idx_b = idx_b[valid]
        if idx_a.size == 0:
            continue
        take = min(needed, idx_a.size)
        idx_a_chunks.append(idx_a[:take])
        idx_b_chunks.append(idx_b[:take])
        needed -= take

    final_a = np.concatenate(idx_a_chunks)
    final_b = np.concatenate(idx_b_chunks)
    return np.sum(embeddings[final_a] * embeddings[final_b], axis=1)


def exact_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    similarity_matrix = embeddings @ embeddings.T
    upper_tri = np.triu_indices(similarity_matrix.shape[0], k=1)
    return similarity_matrix[upper_tri]


def compute_similarity_distribution(
    embeddings: np.ndarray,
    max_exact_pairs: int,
    sample_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_questions = embeddings.shape[0]
    total_pairs = num_questions * (num_questions - 1) // 2

    if total_pairs <= 0:
        return np.array([], dtype=np.float32), {
            "method": "none",
            "total_possible_pairs": int(total_pairs),
            "pairs_used": 0,
        }

    if total_pairs <= max_exact_pairs:
        similarities = exact_pairwise_similarities(embeddings)
        return similarities, {
            "method": "exact",
            "total_possible_pairs": int(total_pairs),
            "pairs_used": int(similarities.size),
        }

    sampled_pairs = min(sample_pairs, total_pairs)
    similarities = sample_pairwise_similarities(embeddings, sampled_pairs, seed)
    return similarities, {
        "method": "sampled",
        "total_possible_pairs": int(total_pairs),
        "pairs_used": int(similarities.size),
    }


def plot_overall_similarity_histogram(
    similarities: np.ndarray,
    output_path: Path,
) -> None:
    if plt is None:
        return
    if similarities.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(similarities, bins=50, color="#1f77b4", edgecolor="white", alpha=0.9)
    plt.title("Overall Question Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Pair Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_question_type_counts(
    counts_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if plt is None:
        return
    if counts_df.empty:
        return

    ordered = counts_df.sort_values("question_count", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(ordered["question_type"], ordered["question_count"], color="#2ca02c")
    plt.title("Question Count by Type")
    plt.xlabel("Question Type")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_within_type_boxplot(
    type_similarity_samples: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    if plt is None:
        return
    non_empty = {k: v for k, v in type_similarity_samples.items() if v.size > 0}
    if not non_empty:
        return

    labels = sorted(non_empty.keys())
    data = [non_empty[label] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title("Within-Type Cosine Similarity Distribution")
    plt.xlabel("Question Type")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def create_tsne_plot(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    max_points: int,
    perplexity: float,
    seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    if TSNE is None or plt is None:
        return {"status": "skipped", "reason": "matplotlib/sklearn not available"}

    total_points = embeddings.shape[0]
    if total_points < 3:
        return {"status": "skipped", "reason": "not enough points for t-SNE"}

    rng = np.random.default_rng(seed)
    if total_points > max_points:
        sample_idx = rng.choice(total_points, size=max_points, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(total_points)

    sampled_embeddings = embeddings[sample_idx]
    sampled_frame = frame.iloc[sample_idx].reset_index(drop=True)

    if sampled_embeddings.shape[0] < 3:
        return {"status": "skipped", "reason": "not enough sampled points for t-SNE"}

    adjusted_perplexity = min(
        perplexity,
        max(2.0, float(sampled_embeddings.shape[0] - 1)),
    )
    if adjusted_perplexity >= sampled_embeddings.shape[0]:
        adjusted_perplexity = float(sampled_embeddings.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=adjusted_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(sampled_embeddings)

    label_series = sampled_frame["question_type"].astype(str)
    unique_labels = sorted(label_series.unique().tolist())
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))

    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(unique_labels):
        mask = label_series == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=16,
            alpha=0.8,
            color=cmap(idx),
            label=label,
        )
    plt.title("t-SNE Projection of Processed Questions (Cosine Metric)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return {
        "status": "ok",
        "points_plotted": int(sampled_embeddings.shape[0]),
        "perplexity_used": float(adjusted_perplexity),
    }


def safe_slug(value: str) -> str:
    slug = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value.lower())
    slug = slug.strip("_")
    return slug or "unknown"


def create_single_type_tsne_plot(
    question_type: str,
    embeddings: np.ndarray,
    max_points: int,
    perplexity: float,
    seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    if TSNE is None or plt is None:
        return {
            "question_type": question_type,
            "status": "skipped",
            "reason": "matplotlib/sklearn not available",
            "output_path": str(output_path),
        }

    total_points = embeddings.shape[0]
    if total_points < 3:
        return {
            "question_type": question_type,
            "status": "skipped",
            "reason": "not enough points for t-SNE",
            "output_path": str(output_path),
        }

    rng = np.random.default_rng(seed)
    if total_points > max_points:
        sample_idx = rng.choice(total_points, size=max_points, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(total_points)

    sampled_embeddings = embeddings[sample_idx]
    if sampled_embeddings.shape[0] < 3:
        return {
            "question_type": question_type,
            "status": "skipped",
            "reason": "not enough sampled points for t-SNE",
            "output_path": str(output_path),
        }

    adjusted_perplexity = min(
        perplexity,
        max(2.0, float(sampled_embeddings.shape[0] - 1)),
    )
    if adjusted_perplexity >= sampled_embeddings.shape[0]:
        adjusted_perplexity = float(sampled_embeddings.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=adjusted_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(sampled_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.85, color="#1f77b4")
    plt.title(f"t-SNE Projection: {question_type} Questions")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return {
        "question_type": question_type,
        "status": "ok",
        "points_plotted": int(sampled_embeddings.shape[0]),
        "perplexity_used": float(adjusted_perplexity),
        "output_path": str(output_path),
    }


def run_analysis(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined, files = load_processed_questions(
        input_dir=args.input_dir,
        file_glob=args.file_glob,
        text_column=args.text_column,
        id_column=args.id_column,
        type_column=args.question_type_column,
    )

    if args.max_questions > 0 and len(combined) > args.max_questions:
        combined = (
            combined.sample(n=args.max_questions, random_state=args.seed)
            .sort_index()
            .reset_index(drop=True)
        )
        print(f"Using sampled subset of {len(combined)} questions (--max-questions).")

    if len(combined) < 2:
        raise ValueError("Need at least two processed questions for similarity analysis.")

    combined_out = args.output_dir / "combined_processed_questions.csv"
    combined.to_csv(combined_out, index=False)

    embeddings = encode_questions(
        texts=combined[args.text_column].tolist(),
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )

    overall_sims, overall_meta = compute_similarity_distribution(
        embeddings=embeddings,
        max_exact_pairs=args.max_exact_pairs,
        sample_pairs=args.overall_sample_pairs,
        seed=args.seed,
    )
    overall_stats = summarize_similarities(overall_sims)

    type_counts = (
        combined["question_type"]
        .value_counts(dropna=False)
        .rename_axis("question_type")
        .reset_index(name="question_count")
        .sort_values("question_count", ascending=False)
        .reset_index(drop=True)
    )
    type_counts_out = args.output_dir / "question_type_counts.csv"
    type_counts.to_csv(type_counts_out, index=False)

    per_type_rows: List[Dict[str, Any]] = []
    type_similarity_samples: Dict[str, np.ndarray] = {}
    unique_types = sorted(combined["question_type"].astype(str).unique().tolist())
    for type_index, question_type in enumerate(tqdm(unique_types, desc="Per-type analysis")):
        subset_idx = np.where(combined["question_type"].to_numpy() == question_type)[0]
        subset_embeddings = embeddings[subset_idx]
        subset_questions = subset_embeddings.shape[0]

        if subset_questions < 2:
            per_type_rows.append(
                {
                    "question_type": question_type,
                    "question_count": int(subset_questions),
                    "similarity_method": "none",
                    "total_possible_pairs": 0,
                    **summarize_similarities(np.array([], dtype=np.float32)),
                }
            )
            type_similarity_samples[question_type] = np.array([], dtype=np.float32)
            continue

        sims, meta = compute_similarity_distribution(
            embeddings=subset_embeddings,
            max_exact_pairs=args.max_exact_pairs,
            sample_pairs=args.per_type_sample_pairs,
            seed=args.seed + type_index * 7919,
        )
        type_similarity_samples[question_type] = sims
        stats = summarize_similarities(sims)
        per_type_rows.append(
            {
                "question_type": question_type,
                "question_count": int(subset_questions),
                "similarity_method": meta["method"],
                "total_possible_pairs": int(meta["total_possible_pairs"]),
                **stats,
            }
        )

    per_type_df = pd.DataFrame(per_type_rows).sort_values(
        "question_count", ascending=False
    )
    per_type_out = args.output_dir / "question_type_similarity_stats.csv"
    per_type_df.to_csv(per_type_out, index=False)

    tsne_info: Dict[str, Any] = {"status": "skipped", "reason": "skip-plots enabled"}
    per_type_tsne_info: List[Dict[str, Any]] = []
    if not args.skip_plots:
        plot_overall_similarity_histogram(
            overall_sims,
            args.output_dir / "overall_similarity_histogram.png",
        )
        plot_question_type_counts(type_counts, args.output_dir / "question_type_counts.png")
        plot_within_type_boxplot(
            type_similarity_samples,
            args.output_dir / "question_type_similarity_boxplot.png",
        )
        tsne_info = create_tsne_plot(
            frame=combined,
            embeddings=embeddings,
            max_points=args.plot_max_points,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
            output_path=args.output_dir / "question_tsne.png",
        )

        per_type_tsne_dir = args.output_dir / "question_tsne_by_type"
        per_type_tsne_dir.mkdir(parents=True, exist_ok=True)
        for type_index, question_type in enumerate(unique_types):
            subset_idx = np.where(combined["question_type"].to_numpy() == question_type)[0]
            subset_embeddings = embeddings[subset_idx]
            out_path = per_type_tsne_dir / f"question_tsne_{safe_slug(question_type)}.png"
            info = create_single_type_tsne_plot(
                question_type=question_type,
                embeddings=subset_embeddings,
                max_points=args.plot_max_points,
                perplexity=args.tsne_perplexity,
                seed=args.seed + type_index * 101,
                output_path=out_path,
            )
            per_type_tsne_info.append(info)
    else:
        for question_type in unique_types:
            per_type_tsne_info.append(
                {
                    "question_type": question_type,
                    "status": "skipped",
                    "reason": "skip-plots enabled",
                }
            )

    summary = {
        "num_files_loaded": len(files),
        "files_loaded": [path.name for path in files],
        "num_questions_analyzed": int(len(combined)),
        "embedding_provider": "openrouter",
        "embedding_model": args.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "overall_similarity": {
            "method": overall_meta["method"],
            "total_possible_pairs": overall_meta["total_possible_pairs"],
            **overall_stats,
        },
        "question_type_breakdown": per_type_rows,
        "tsne": tsne_info,
        "tsne_by_type": per_type_tsne_info,
        "output_files": {
            "combined_questions_csv": str(combined_out),
            "question_type_counts_csv": str(type_counts_out),
            "question_type_similarity_csv": str(per_type_out),
            "overall_histogram_png": str(
                args.output_dir / "overall_similarity_histogram.png"
            ),
            "question_type_counts_png": str(args.output_dir / "question_type_counts.png"),
            "question_type_boxplot_png": str(
                args.output_dir / "question_type_similarity_boxplot.png"
            ),
            "tsne_png": str(args.output_dir / "question_tsne.png"),
            "tsne_by_type_dir": str(args.output_dir / "question_tsne_by_type"),
        },
    }

    summary_path = args.output_dir / "similarity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nAnalysis complete.")
    print(f"Questions analyzed: {len(combined)}")
    print(f"Overall similarity method: {overall_meta['method']}")
    print(f"Overall mean cosine similarity: {overall_stats['mean']}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
