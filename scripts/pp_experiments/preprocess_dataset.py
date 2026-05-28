"""
Preprocess the raw pairwise-preference + answer-metadata files into a
feature matrix ready for classifier training.

Usage::
    python preprocess_dataset.py --dataset bbh --judge openai/gpt-oss-20b
    python preprocess_dataset.py --dataset-dir dataset/bbh__openai_gpt-oss-20b

Reads from:
  dataset/<dataset>/<judge>/pairwise_preferences.csv
  dataset/<dataset>/answer_metadata.csv

Writes:
  dataset/<dataset>/<judge>/features.csv

Each row in features.csv corresponds to one pairwise comparison and contains:
  question_id, model_a, model_b, preference,
  diff_token_len,
  diff_header_count, diff_list_count, diff_bold_count
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import get_dataset_dir, get_dataset_judge_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw dataset files into a feature matrix."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", metavar="DATASET",
        help="Dataset name (e.g. bbh). Must be combined with --judge.",
    )
    parser.add_argument(
        "--judge", metavar="JUDGE",
        help="Judge model ID (e.g. openai/gpt-oss-20b). Required when using --dataset.",
    )
    args = parser.parse_args()
    if args.dataset and not args.judge:
        parser.error("--judge is required when using --dataset.")
    return args

# Columns in answer_metadata that we want to compute A-minus-B differences for.
# Each entry is (output_column_name, source_column_in_metadata).
DIFF_FEATURES = [
    ("diff_token_len",    "token_len"),
    ("diff_header_count", "header_count__total"),
    ("diff_list_count",   "list_count__total"),
    ("diff_bold_count",   "bold_count__total"),
    ("diff_math_count",   "math_count__total"),
    ("diff_echo",        "is_echo"), # 1 if A is ECHO, -1 if B is ECHO, 0 if neither or both are ECHO
    ("diff_correct",     "is_correct"), # 1 if A is correct, -1 if B is correct, 0 if neither or both are correct
]

# XOR features: The output column will be 1 if exactly one of the models has the feature, and 0 if both or neither have it.
# Each entry is (output_column_name, source_column_in_metadata).
XOR_FEATURES = [
    ("one_correct", "is_correct"), # 1 if exactly one of the models is correct, 0 if both or neither are correct
]

# Composite features to compute from the basic diff features. Each entry is
# (output_column_name, list_of_input_diff_columns). The input diff columns must be present in DIFF_FEATURES.
COMPOSITE_FEATURES = [
    ("diff_md_count", ["diff_header_count", "diff_list_count", "diff_bold_count"]),
]


def main():
    args = parse_args()
    dataset_dir = get_dataset_dir(args.dataset)
    preferences_dir = get_dataset_judge_dir(args.dataset, args.judge)

    if not dataset_dir.exists() or not preferences_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {preferences_dir}. "
            "Run build_dataset.py first."
        )

    meta_path  = dataset_dir / "answer_metadata.csv"
    prefs_path = preferences_dir / "pairwise_preferences.csv"
    out_path   = preferences_dir / "features.csv"

    prefs = pd.read_csv(prefs_path, dtype={"question_id": str})
    meta  = pd.read_csv(meta_path,  dtype={"question_id": str})

    print(f"Loaded {len(prefs)} pairwise preferences.")
    print(f"Loaded {len(meta)} answer-metadata rows.")

    # Index metadata for fast lookup: (model_id, question_id) → row
    meta_indexed = meta.set_index(["model_id", "question_id"])

    rows = []
    n_missing = 0
    for _, pref in prefs.iterrows():
        qid     = pref["question_id"]
        model_a = pref["model_a"]
        model_b = pref["model_b"]
        key_a   = (model_a, qid)
        key_b   = (model_b, qid)

        if key_a not in meta_indexed.index or key_b not in meta_indexed.index:
            n_missing += 1
            continue

        row_a = meta_indexed.loc[key_a]
        row_b = meta_indexed.loc[key_b]

        feature_row = {
            "question_id": qid,
            "model_a":     model_a,
            "model_b":     model_b,
            "preference":  int(pref["preference"]),
        }
        for out_col, src_col in DIFF_FEATURES:
            feature_row[out_col] = row_a[src_col] - row_b[src_col]

        for out_col, src_col in XOR_FEATURES:
            feature_row[out_col] = int(row_a[src_col] != row_b[src_col])

        for out_col, input_cols in COMPOSITE_FEATURES:
            feature_row[out_col] = sum(feature_row[in_col] for in_col in input_cols)

        rows.append(feature_row)

    if n_missing:
        print(f"Warning: skipped {n_missing} preferences with missing metadata.")

    out_cols = [out for out, _ in DIFF_FEATURES] + [out for out, _ in XOR_FEATURES] + [out for out, _ in COMPOSITE_FEATURES]

    features = pd.DataFrame(rows, columns=["question_id", "model_a", "model_b", "preference"] + out_cols)
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features)} feature rows → {out_path}")
    print(features.describe())

    # Correlation heatmap over feature columns (including preference)
    heatmap_cols = ["preference"] + out_cols
    corr = features[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(len(heatmap_cols), len(heatmap_cols) - 1))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()

    heatmap_path = preferences_dir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap → {heatmap_path}")

    # Correlation dendrogram using only feature columns (excluding preference, xor and composite features)
    from scipy.cluster.hierarchy import linkage, dendrogram, distance
    dendrogram_cols = [out for out, _ in DIFF_FEATURES]
    corr = features[dendrogram_cols].corr("spearman")
    # For hierarchical clustering, we need a distance matrix, so we convert correlation to distance by using (1 - corr).
    # Squareform is needed because linkage expects a condensed distance matrix.
    # Squareform: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
    # Linkage: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    linked = linkage(distance.squareform(1 - corr), method="complete")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linked, labels=dendrogram_cols, ax=ax, orientation="left")
    plt.xlabel("Dissimilarity (1 - |Spearman Rho|)")
    plt.ylabel("Features")
    ax.set_title("Feature correlation dendrogram")
    fig.tight_layout()
    dendrogram_path = preferences_dir / "correlation_dendrogram.png"
    fig.savefig(dendrogram_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation dendrogram → {dendrogram_path}")


if __name__ == "__main__":
    main()
