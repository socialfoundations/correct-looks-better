"""
Rank-correlation table: pairwise (BradleyTerry) and AM-no-GT vs. gold standard.

For each judge in JUDGES, computes Spearman's rho (default) between:
  - Pairwise ranking (BradleyTerry by default, configurable via --methods)
  - AM-no-GT ranking
and the accuracy-based gold-standard ranking, for each configured dataset.

Run from repo root:
    export PYTHONPATH=scripts
    python scripts/judge_comparison/judge_comparison_table.py [--datasets ...] [--methods ...] [--metric rho|tau|r]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import pandas as pd

from utils import (
    data_configs,
    load_accuracy_ranking,
    load_model_rankings,
    load_llm_judge_ranking,
    rank_similarity,
)

# ── Configuration ─────────────────────────────────────────────────────────────

JUDGES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/o3",
    "microsoft/phi-4",
    "google/gemma-3-27b-it"
]

DEFAULT_DATASETS = ["mmlu_pro", "gpqa_diamond", "simple_qa", "gsm8k", "bbh"]
DEFAULT_METHODS = ["BradleyTerry", "Direct Judge"]

JUDGE_LATEX_NAMES = {
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/o3": "o3",
    "microsoft/phi-4": "phi-4",
    "microsoft/Phi-4-reasoning": "phi-4-reasoning",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
}

METHOD_LATEX_NAMES = {
    "BradleyTerry": "BT",
    "TrueSkill-M": "TS-M",
    "TrueSkill-MA": "TS-MA",
    "WinRate": "WR",
    "Direct Judge": "DJ",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _judge_short(judge: str) -> str:
    return judge.split("/")[-1]


def _load_gt(dataset: str):
    return load_accuracy_ranking(dataset)


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


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(datasets: list, judges: list, methods: list, metric: str) -> pd.DataFrame:
    methods = methods or []
    col_tuples = []
    for judge in judges:
        j = _judge_short(judge)
        for m in methods:
            col_tuples.append((j, m))

    columns = pd.MultiIndex.from_tuples(col_tuples, names=["judge", "method"])
    df = pd.DataFrame(index=datasets, columns=columns, dtype=float)
    df.index.name = "dataset"

    for dataset in datasets:
        print(f"  {dataset}...", flush=True)
        try:
            gt_rank = _load_gt(dataset)
        except Exception as e:
            print(f"    [error] could not load gold standard: {e}")
            continue

        for judge in judges:
            j = _judge_short(judge)

            for method in methods:
                try:
                    if method == "Direct Judge":
                        rank = load_llm_judge_ranking(dataset, judge_model=judge)
                    else:
                        rank = _load_pairwise(dataset, method, judge)
                    df.loc[dataset, (j, method)] = _corr(rank, gt_rank, metric)
                except Exception as e:
                    print(f"    [warn] {j} / {method}: {e}")

    return df


# ── Output formatters ─────────────────────────────────────────────────────────

METRIC_NAMES = {
    "rho": "Spearman's rho",
    "tau": "Kendall's tau",
    "r": "Pearson's r",
}


def to_latex(df, metric, methods, judges) -> str:
    methods = methods or []
    judges_short = [_judge_short(j) for j in judges]
    judges_tex = [JUDGE_LATEX_NAMES.get(_judge_short(j), _judge_short(j)) for j in judges]
    methods_tex = [METHOD_LATEX_NAMES.get(m, m) for m in methods]
    n_cols = len(judges_short) * len(methods)

    col_groups = " & ".join(
        f"\\multicolumn{{{len(methods)}}}{{c}}{{{j}}}" for j in judges_tex
    )
    col_headers = " & ".join(methods_tex * len(judges_short))
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * len(methods)}-{1 + (i + 1) * len(methods)}}}"
        for i in range(len(judges_short))
    )

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * n_cols}}}",
        "\\toprule",
        f" & {col_groups} \\\\",
        cmidrules,
        f" & {col_headers} \\\\",
        "\\midrule",
    ]

    for dataset, row in df.iterrows():
        cells = []
        for judge in judges_short:
            for method in methods:
                v = row[(judge, method)]
                is_nan = isinstance(v, float) and v != v
                if is_nan:
                    cells.append(("--", judge, method, float("-inf")))
                else:
                    cells.append((f"{v:.2f}", judge, method, v))

        formatted = []
        for judge in judges_short:
            group = [(s, val) for s, j, _, val in cells if j == judge]
            best = max(val for _, val in group)
            for s, val in group:
                if len(methods) > 1 and val == best and val != float("-inf"):
                    formatted.append(f"\\textbf{{{s}}}")
                else:
                    formatted.append(s)

        dataset_tex = dataset.replace("_", r"\_")
        lines.append(f"{dataset_tex} & {' & '.join(formatted)} \\\\")

    lines.append("\\midrule")
    mean_std = []
    for judge in judges_short:
        for method in methods:
            col = df[(judge, method)].dropna().values
            if len(col) == 0:
                mean_std.append("--")
            else:
                mean_std.append(f"{np.mean(col):.2f} $\\pm$ {np.std(col):.2f}")
    lines.append(f"Mean $\\pm$ Std & {' & '.join(mean_std)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{METRIC_NAMES[metric]} vs.\\ accuracy gold standard.}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def to_markdown(df, metric, methods, judges) -> str:
    methods = methods or []
    judges_short = [_judge_short(j) for j in judges]
    judges_display = [JUDGE_LATEX_NAMES.get(_judge_short(j), _judge_short(j)) for j in judges]
    methods_display = [METHOD_LATEX_NAMES.get(m, m) for m in methods]

    # Flatten multi-level columns as "judge / method"
    flat_headers = [
        f"{j} / {m}" for j in judges_display for m in methods_display
    ]
    header = "| Dataset | " + " | ".join(flat_headers) + " |"
    sep = "| :--- | " + " | ".join(["---:"] * len(flat_headers)) + " |"

    lines = [
        f"## {METRIC_NAMES[metric]} vs. gold-standard accuracy ranking",
        "",
        header,
        sep,
    ]

    for dataset, row in df.iterrows():
        formatted = []
        for judge in judges_short:
            group_vals = []
            for method in methods:
                v = row[(judge, method)]
                is_nan = isinstance(v, float) and v != v
                group_vals.append(float("-inf") if is_nan else v)
            best = max(group_vals)
            for v in group_vals:
                is_nan = v == float("-inf")
                if is_nan:
                    formatted.append("--")
                elif len(methods) > 1 and v == best:
                    formatted.append(f"**{v:.2f}**")
                else:
                    formatted.append(f"{v:.2f}")

        lines.append(f"| {dataset} | " + " | ".join(formatted) + " |")

    mean_std = []
    for judge in judges_short:
        for method in methods:
            col = df[(judge, method)].dropna().values
            mean_std.append("--" if len(col) == 0 else f"{np.mean(col):.2f} ± {np.std(col):.2f}")
    lines.append(f"| **Mean ± Std** | " + " | ".join(mean_std) + " |")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to include: pairwise (BradleyTerry, TrueSkill-M, WinRate) and/or Direct Judge",
    )
    parser.add_argument(
        "--metric",
        default="tau",
        choices=["rho", "tau", "r"],
        help="Rank-correlation metric",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table.")
    parser.add_argument("--markdown", action="store_true", help="Output Markdown table.")
    args = parser.parse_args()

    print(f"\nBuilding table ({METRIC_NAMES[args.metric]} vs. gold standard)...\n")
    df = build_table(args.datasets, JUDGES, args.methods, args.metric)

    if args.latex:
        print(to_latex(df, args.metric, args.methods, JUDGES))
    elif args.markdown:
        print(to_markdown(df, args.metric, args.methods, JUDGES))
    else:
        print(f"\n{METRIC_NAMES[args.metric]} vs. gold standard\n")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


if __name__ == "__main__":
    main()
