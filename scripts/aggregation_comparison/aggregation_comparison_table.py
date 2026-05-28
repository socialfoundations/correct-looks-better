"""
Rank-correlation table: WinRate, Elo, BradleyTerry, TrueSkill-M vs. gold standard.

For a fixed judge model, computes the chosen correlation metric between each
aggregation method's ranking and the accuracy-based gold-standard ranking,
for each configured dataset.

Run from repo root:
    export PYTHONPATH=scripts
    python scripts/aggregation_comparison/aggregation_comparison_table.py [--judge ...] [--metric tau|rho|r] [--latex] [--markdown]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from utils import (
    data_configs,
    load_accuracy_ranking,
    load_model_rankings,
    rank_similarity,
)

DEFAULT_DATASETS = ["mmlu_pro", "gpqa_diamond", "simple_qa", "gsm8k", "bbh"]
DEFAULT_METHODS = ["WinRate", "BradleyTerry", "Elo", "TrueSkill-M"]
DEFAULT_JUDGE = "openai/gpt-oss-20b"

METRIC_NAMES = {
    "tau": "Kendall's tau",
    "rho": "Spearman's rho",
    "r": "Pearson's r",
}

METHOD_LATEX_NAMES = {
    "WinRate": "WinRate",
    "Elo": "Elo",
    "BradleyTerry": "BT",
    "TrueSkill-M": "TrueSkill",
}

DATASET_LATEX_NAMES = {
    "mmlu_pro": "MMLU-Pro",
    "gpqa_diamond": "GPQA-Diamond",
    "simple_qa": "SimpleQA",
    "gsm8k": "GSM8K",
    "bbh": "BBH",
}


def _corr(rank, gt_rank, metric: str) -> float:
    rank_s = sorted(rank, key=lambda x: x[0])
    gt_s = sorted(gt_rank, key=lambda x: x[0])
    scores = [s for _, s in rank_s]
    gt_scores = [s for _, s in gt_s]
    c, _ = rank_similarity(scores, gt_scores, method=metric)
    return round(c, 4)


def _load_pairwise(dataset: str, method: str, judge: str) -> list:
    n_all = data_configs[dataset]["n_pairwise_comps"].get(method)
    return load_model_rankings(
        dataset=dataset,
        method=method,
        judge_model=judge,
        shuffle=False,
        with_replacement=False,
        n=n_all,
    )


def build_table(datasets, methods, judge, metric) -> pd.DataFrame:
    df = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    df.index.name = "dataset"

    for dataset in datasets:
        print(f"  {dataset}...", flush=True)
        try:
            gt_rank = load_accuracy_ranking(dataset)
        except Exception as e:
            print(f"    [error] gold standard: {e}", file=sys.stderr)
            continue

        for method in methods:
            try:
                rank = _load_pairwise(dataset, method, judge)
                df.loc[dataset, method] = _corr(rank, gt_rank, metric)
            except Exception as e:
                print(f"    [warn] {method}: {e}", file=sys.stderr)

    return df


def to_latex(df, metric, judge) -> str:
    methods = list(df.columns)
    col_headers = " & ".join(METHOD_LATEX_NAMES.get(m, m) for m in methods)
    n_cols = len(methods)
    judge_display = judge.split("/")[-1]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * n_cols}}}",
        "\\toprule",
        f"Dataset & {col_headers} \\\\",
        "\\midrule",
    ]

    for dataset, row in df.iterrows():
        vals = [row[m] for m in methods]
        non_nan = [v for v in vals if not (isinstance(v, float) and v != v)]
        best = max(non_nan) if non_nan else None

        formatted = []
        for v in vals:
            is_nan = isinstance(v, float) and v != v
            if is_nan:
                formatted.append("--")
            elif best is not None and v == best:
                formatted.append(f"\\textbf{{{v:.4f}}}")
            else:
                formatted.append(f"{v:.4f}")

        dataset_tex = DATASET_LATEX_NAMES.get(dataset, dataset.replace("_", r"\_"))
        lines.append(f"{dataset_tex} & {' & '.join(formatted)} \\\\")

    lines.append("\\midrule")
    means, stds = [], []
    for method in methods:
        col = df[method].dropna().values
        if len(col) == 0:
            means.append("--")
            stds.append("--")
        else:
            means.append(f"{np.mean(col):.4f}")
            stds.append(f"{np.std(col):.4f}")
    lines.append(f"Mean & {' & '.join(means)} \\\\")
    lines.append(f"Std & {' & '.join(stds)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{METRIC_NAMES[metric]} between rank aggregation methods and accuracy gold standard (judge: {judge_display}).}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def to_markdown(df, metric, judge) -> str:
    methods = list(df.columns)
    judge_display = judge.split("/")[-1]

    display_names = [METHOD_LATEX_NAMES.get(m, m) for m in methods]
    header = "| Dataset | " + " | ".join(display_names) + " |"
    sep = "| :--- | " + " | ".join(["---:"] * len(methods)) + " |"

    lines = [
        f"## {METRIC_NAMES[metric]} vs. gold-standard accuracy ranking",
        f"*Judge: {judge_display}*",
        "",
        header,
        sep,
    ]

    for dataset, row in df.iterrows():
        vals = [row[m] for m in methods]
        non_nan = [v for v in vals if not (isinstance(v, float) and v != v)]
        best = max(non_nan) if non_nan else None

        formatted = []
        for v in vals:
            is_nan = isinstance(v, float) and v != v
            if is_nan:
                formatted.append("--")
            elif best is not None and v == best:
                formatted.append(f"**{v:.4f}**")
            else:
                formatted.append(f"{v:.4f}")

        lines.append(f"| {dataset} | " + " | ".join(formatted) + " |")

    means = []
    for method in methods:
        col = df[method].dropna().values
        means.append("--" if len(col) == 0 else f"{np.mean(col):.4f}")
    lines.append(f"| **Mean** | " + " | ".join(means) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare rank aggregation methods vs. accuracy gold standard."
    )
    parser.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        help="Judge model that provided pairwise preferences.",
    )
    parser.add_argument(
        "--metric",
        default="tau",
        choices=["tau", "rho", "r"],
        help="Rank-correlation metric.",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table.")
    parser.add_argument("--markdown", action="store_true", help="Output Markdown table.")
    args = parser.parse_args()

    print(
        f"\nBuilding table ({METRIC_NAMES[args.metric]} vs. gold standard, judge={args.judge})...\n"
    )
    df = build_table(DEFAULT_DATASETS, DEFAULT_METHODS, args.judge, args.metric)

    if args.latex:
        print(to_latex(df, args.metric, args.judge))
    elif args.markdown:
        print(to_markdown(df, args.metric, args.judge))
    else:
        print(f"\n{METRIC_NAMES[args.metric]} vs. gold standard (judge: {args.judge})\n")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


if __name__ == "__main__":
    main()
