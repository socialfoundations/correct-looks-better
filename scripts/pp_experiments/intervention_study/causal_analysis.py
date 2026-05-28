"""
Causal analysis of echo intervention results.

Reads intervention_raw.csv for one or more judges and computes:
  - P(A>B | echo=x) for x in {-1, 0, +1}
  - Stratified by one_correct
  - Bootstrap CIs on pairwise contrasts

Cross-judge comparison summary is appended when multiple judges are provided.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_Z95 = 1.96  # z-score for 95% CI


# ---------------------------------------------------------------------------
# Wald confidence interval helpers
# ---------------------------------------------------------------------------

def wald_rate(labels: np.ndarray) -> Tuple[float, float, float]:
    """Wald 95% CI for a binary proportion. Returns (point_estimate, ci_low, ci_high)."""
    n = len(labels)
    p = float(labels.mean())
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return p, p - _Z95 * se, p + _Z95 * se


def wald_rate_diff(a_labels: np.ndarray, b_labels: np.ndarray) -> Tuple[float, float, float]:
    """Wald 95% CI for difference in proportions: mean(a) - mean(b). Returns (point, ci_low, ci_high)."""
    p_a = float(a_labels.mean())
    p_b = float(b_labels.mean())
    diff = p_a - p_b
    se = np.sqrt(p_a * (1 - p_a) / len(a_labels) + p_b * (1 - p_b) / len(b_labels))
    return diff, diff - _Z95 * se, diff + _Z95 * se


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

ECHO_LABELS = {1: "echo=+1 (A echoes)", 0: "echo= 0 (neutral)", -1: "echo=-1 (B echoes)"}
CONTRASTS = [
    (1,  0, "echo=+1 vs echo= 0"),
    (-1, 0, "echo=-1 vs echo= 0"),
    (1, -1, "echo=+1 vs echo=-1"),
]


def _rate_row(label: str, subset: pd.DataFrame) -> str:
    if len(subset) == 0:
        return f"  {label:<35}  n=0  (no data)"
    vals = subset["preference"].dropna().values.astype(float)
    rate, lo, hi = wald_rate(vals)
    return (
        f"  {label:<35}  n={len(vals):>4}  "
        f"P(A>B)={rate:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]"
    )


def _contrast_row(label: str, a_vals: np.ndarray, b_vals: np.ndarray) -> str:
    if len(a_vals) == 0 or len(b_vals) == 0:
        return f"  {label:<40}  n/a"
    eff, lo, hi = wald_rate_diff(a_vals, b_vals)
    return (
        f"  {label:<40}  "
        f"effect={eff:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"
    )


def analyze_judge(df: pd.DataFrame, judge_id: str) -> List[str]:
    """Produce result lines for a single judge's intervention_raw.csv."""
    lines: List[str] = []
    lines.append(f"Judge: {judge_id}")
    lines.append(f"Total rows: {len(df)}  (usable: {df['preference'].notna().sum()})")

    df = df.dropna(subset=["preference"]).copy()
    df["preference"] = df["preference"].astype(int)

    # ------------------------------------------------------------------
    # 1. Overall P(A>B | echo=x)
    # ------------------------------------------------------------------
    lines.append("\n[Overall P(A>B) by echo condition]")
    echo_data: dict[int, np.ndarray] = {}
    for ec in [-1, 0, 1]:
        sub = df[df["echo_condition"] == ec]
        vals = sub["preference"].values.astype(float)
        echo_data[ec] = vals
        lines.append(_rate_row(ECHO_LABELS[ec], sub))

    lines.append("\n[Contrasts — effect of echo on P(A>B)]")
    for ec_a, ec_b, label in CONTRASTS:
        lines.append(_contrast_row(label, echo_data[ec_a], echo_data[ec_b]))

    # ------------------------------------------------------------------
    # 2. Stratified by one_correct
    # ------------------------------------------------------------------
    lines.append("\n[Stratified by one_correct]")
    for oc in sorted(df["one_correct"].unique()):
        stratum = df[df["one_correct"] == oc]
        label = "one_correct=1 (verifiable)" if oc == 1 else "one_correct=0 (both/neither correct)"
        lines.append(f"\n  {label}  (n={len(stratum)})")
        oc_echo_data: dict[int, np.ndarray] = {}
        for ec in [-1, 0, 1]:
            sub = stratum[stratum["echo_condition"] == ec]
            vals = sub["preference"].values.astype(float)
            oc_echo_data[ec] = vals
            lines.append("  " + _rate_row(ECHO_LABELS[ec], sub))
        lines.append(f"  Contrasts:")
        for ec_a, ec_b, clabel in CONTRASTS:
            lines.append("  " + _contrast_row(clabel, oc_echo_data[ec_a], oc_echo_data[ec_b]))

    return lines


def cross_judge_summary(
    judge_rates: dict[str, dict[int, float]],
) -> List[str]:
    """Produce a summary table comparing P(A>B | echo=x) across judges."""
    lines = ["\n" + "=" * 70, "Cross-judge summary: P(A>B | echo=x)", "=" * 70]
    header = f"  {'Judge':<45}" + "".join(f"  {ECHO_LABELS[ec][:10]}" for ec in [-1, 0, 1])
    lines.append(header)
    for judge_id, rates in sorted(judge_rates.items()):
        row = f"  {judge_id:<45}"
        for ec in [-1, 0, 1]:
            v = rates.get(ec)
            row += f"  {v:.3f}     " if v is not None else "  n/a       "
        lines.append(row)
    return lines


def run_analysis(results_dirs: dict[str, Path]) -> None:
    """Load intervention_raw.csv for each judge, run analysis, save results.

    Args:
        results_dirs: {judge_id: path_to_judge_results_dir}
    """
    judge_rates: dict[str, dict[int, float]] = {}
    all_lines: List[str] = []

    for judge_id, rdir in results_dirs.items():
        raw_path = rdir / "intervention_raw.csv"
        if not raw_path.exists():
            print(f"[WARN] No intervention_raw.csv found for judge {judge_id} at {raw_path}")
            continue

        df = pd.read_csv(raw_path, dtype={"question_id": str})
        print(f"\nAnalysing {judge_id}: {len(df)} rows loaded.")

        sep = "=" * 70
        all_lines += [sep, f"RESULTS — {judge_id}", sep]
        judge_lines = analyze_judge(df, judge_id)
        all_lines += judge_lines

        # Collect overall rates for cross-judge table
        df_clean = df.dropna(subset=["preference"]).copy()
        df_clean["preference"] = df_clean["preference"].astype(int)
        judge_rates[judge_id] = {
            ec: df_clean[df_clean["echo_condition"] == ec]["preference"].mean()
            for ec in [-1, 0, 1]
            if len(df_clean[df_clean["echo_condition"] == ec]) > 0
        }

        # Save per-judge results
        out_path = rdir / "intervention_results.txt"
        out_path.write_text("\n".join(judge_lines) + "\n")
        print(f"Saved results → {out_path}")

    # Cross-judge summary (only meaningful with multiple judges)
    if len(judge_rates) > 1:
        summary_lines = cross_judge_summary(judge_rates)
        all_lines += summary_lines

        # Save to the parent results dir (one level up from any judge dir)
        sample_dir = next(iter(results_dirs.values()))
        summary_path = sample_dir.parent / "intervention_cross_judge_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n")
        print(f"\nSaved cross-judge summary → {summary_path}")

    print("\n" + "\n".join(all_lines))
