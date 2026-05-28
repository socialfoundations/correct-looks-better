"""
Run pairwise judge on intervention records.

For records that need a new judge call (needs_judge=True), queries the LLM judge
and records the preference. Records with needs_judge=False (original condition)
already have their preference populated and are passed through unchanged.

Saves results to: results/{dataset}/{judge}/intervention_raw.csv
"""

import json
from pathlib import Path
from typing import List

import pandas as pd

from rank_no_eval.client import LiteLLMClient
from rank_no_eval.query import PairwiseComparisonQueries


def run_judge(
    records: List[dict],
    judge_id: str,
    client_type: str = "litellm",
) -> List[dict]:
    """Query the judge for records that need it; pass through the rest.

    Args:
        records: Intervention records from interventions.build_interventions().
                 Each must have: question, answer_a, answer_b, needs_judge, preference.
        judge_id: Model ID string (e.g. "openai/o3").
        client_type: "litellm" (default) or "vllm".

    Returns:
        Records with 'preference' populated for all rows (int 1/0, or None on error).
    """
    to_judge = [(i, r) for i, r in enumerate(records) if r["needs_judge"]]

    if to_judge:
        query_method = PairwiseComparisonQueries(variant="a_a")
        if client_type == "vllm":
            from rank_no_eval.client import VLLMClient
            client = VLLMClient(judge_id, query_method)
        else:
            client = LiteLLMClient(judge_id, query_method)

        judge_samples = [
            {
                "p1": {
                    "question": r["question"],
                    "answer": r["answer_a"],
                    "model_id": r["model_a"],
                    "question_id": r["question_id"],
                },
                "p2": {
                    "question": r["question"],
                    "answer": r["answer_b"],
                    "model_id": r["model_b"],
                    "question_id": r["question_id"],
                },
            }
            for _, r in to_judge
        ]

        responses = client.collect_model_answers(judge_samples)

        for (orig_idx, rec), resp in zip(to_judge, responses):
            try:
                winner = json.loads(resp)["winner"]
                preference = 1 if int(winner) == 1 else 0
            except (json.JSONDecodeError, KeyError, ValueError):
                preference = None
            records[orig_idx] = {**rec, "preference": preference}

    return records


def save_results(records: List[dict], out_path: Path) -> None:
    """Save intervention results CSV (metadata only, no raw answer text)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    SAVE_COLS = [
        "question_id", "model_a", "model_b",
        "starting_echo", "echo_condition", "intervention", "n_reps",
        "one_correct", "preference",
    ]
    rows = [{c: r.get(c) for c in SAVE_COLS} for r in records]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
