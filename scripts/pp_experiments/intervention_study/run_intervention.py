"""
Causal Analysis by Intervention on Echo
========================================

Orchestrates the full interventional study pipeline for one intervention type:

  add_echo    — sample pairs where neither answer echoes (echo_a=0, echo_b=0);
                inject echo into A or B to produce counterfactuals.
  remove_echo — sample pairs where both answers echo (echo_a=1, echo_b=1);
                strip echo from A or B to produce counterfactuals.

Pipeline:
  1. Load sample pool from features.csv + answer_metadata.csv, filtered to the
     relevant sub-case for the chosen --intervention.
  2. Sample ~N pairs.
  3. Load raw answer texts via ModelAnswerStore.
  4. For remove_echo: run remove_echo() LLM calls to get clean answers.
  5. Build intervention records (original + 2 counterfactuals per sample).
  6. Run judge on counterfactual conditions; reuse cached original preferences.
  7. Save results/<dataset>__<judge>/<intervention>/intervention_raw.csv.
  8. Run causal analysis and save results.

Run from the repo root::

    python scripts/pp_experiments/intervention_study/run_intervention.py \\
        --intervention add_echo \\
        --dataset bbh \\
        --judge openai/o3 \\
        --n 500 \\
        --seed 42

Run a quick smoke test with::

    python scripts/pp_experiments/intervention_study/run_intervention.py \\
        --intervention add_echo --n 10
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

from utils import get_dataset_dir, get_dataset_judge_dir, get_results_dir, load_env

load_env()

from config import DEFAULT_DATASET, DEFAULT_JUDGE, DEFAULT_N, DEFAULT_SEED
from interventions import build_interventions, EchoExtractionQuery
from run_judge import run_judge, save_results
from causal_analysis import run_analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal analysis of echo via interventions on BBH answer pairs."
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--judge", default=DEFAULT_JUDGE,
        help=f"Judge model ID (default: {DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Total number of samples to draw (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--intervention", choices=["add_echo", "remove_echo"], required=True,
        help="Which intervention to run: add_echo (neither-echo pairs) or remove_echo (both-echo pairs)",
    )
    parser.add_argument(
        "--echo-extractor", default=None,
        help="Model ID for remove_echo() calls (default: same as --judge)",
    )
    parser.add_argument(
        "--client", choices=["litellm", "vllm"], default="litellm",
        help="Client to use for judge calls: litellm (default) or vllm",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def load_sample_pool(dataset: str, judge: str) -> pd.DataFrame:
    """Load features.csv and join with answer_metadata.csv to get per-model echo flags.

    Returns a DataFrame with all features.csv columns plus echo_a and echo_b
    (individual is_echo values for model_a and model_b respectively).
    Filtered to diff_echo=0 rows only.
    """
    features_path = get_dataset_judge_dir(dataset, judge) / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"features.csv not found at {features_path}.\n"
            f"Run preprocess_dataset.py --dataset {dataset} --judge {judge} first."
        )

    meta_path = get_dataset_dir(dataset) / "answer_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"answer_metadata.csv not found at {meta_path}.")

    pool = pd.read_csv(features_path, dtype={"question_id": str})
    pool = pool.rename(columns={"preference": "original_preference"})
    meta = pd.read_csv(meta_path, dtype={"question_id": str})

    # Join echo_a (is_echo for model_a) and echo_b (is_echo for model_b)
    echo_a = meta[["model_id", "question_id", "is_echo"]].rename(
        columns={"model_id": "model_a", "is_echo": "echo_a"}
    )
    echo_b = meta[["model_id", "question_id", "is_echo"]].rename(
        columns={"model_id": "model_b", "is_echo": "echo_b"}
    )
    pool = pool.merge(echo_a, on=["model_a", "question_id"], how="left")
    pool = pool.merge(echo_b, on=["model_b", "question_id"], how="left")

    # Keep only diff_echo=0 pairs
    pool = pool[pool["diff_echo"] == 0].reset_index(drop=True)

    n_neither = ((pool["echo_a"] == 0) & (pool["echo_b"] == 0)).sum()
    n_both    = ((pool["echo_a"] == 1) & (pool["echo_b"] == 1)).sum()
    print(f"Sample pool (diff_echo=0): {len(pool)} pairs  "
          f"(neither echoes: {n_neither}, both echo: {n_both})")
    return pool


def filter_pool_by_intervention(pool: pd.DataFrame, intervention: str) -> pd.DataFrame:
    """Keep only the sub-case relevant to the chosen intervention."""
    if intervention == "add_echo":
        filtered = pool[(pool["echo_a"] == 0) & (pool["echo_b"] == 0)]
        print(f"Filtered to neither-echo pairs (add_echo): {len(filtered)} pairs")
    else:
        filtered = pool[(pool["echo_a"] == 1) & (pool["echo_b"] == 1)]
        print(f"Filtered to both-echo pairs (remove_echo): {len(filtered)} pairs")
    return filtered.reset_index(drop=True)


def sample_echo_zero(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample n pairs from the (already filtered) diff_echo=0 pool."""
    k = min(n, len(pool))
    if k < n:
        print(f"[WARN] Only {k} diff_echo=0 pairs available; requested {n}.")
    result = pool.sample(n=k, random_state=int(rng.integers(1 << 31)))

    n_neither = ((result["echo_a"] == 0) & (result["echo_b"] == 0)).sum()
    n_both    = ((result["echo_a"] == 1) & (result["echo_b"] == 1)).sum()
    print(f"Sampled {len(result)} pairs  "
          f"(neither echoes: {n_neither}, both echo: {n_both})")
    return result


# ---------------------------------------------------------------------------
# Answer loading
# ---------------------------------------------------------------------------

def load_answer_stores(model_ids: List[str], dataset_enum) -> dict:
    """Load ModelAnswerStore for each model. Returns {model_id: store}."""
    from rank_no_eval.io import load_freeform_dataset
    from rank_no_eval.model_answer import ModelAnswerStore
    from rank_no_eval.utils import QueryMode

    print(f"Loading dataset {dataset_enum.value} from HuggingFace...")
    preloaded = load_freeform_dataset(dataset=dataset_enum)

    stores = {}
    for mid in model_ids:
        print(f"  Loading answers for {mid}...")
        stores[mid] = ModelAnswerStore(
            model_id=mid,
            preloaded_dataset=preloaded,
            dataset=dataset_enum,
            query_mode=QueryMode.FREEFORM,
            with_metadata=False,
        )
    return stores


# ---------------------------------------------------------------------------
# Remove-echo preprocessing
# ---------------------------------------------------------------------------

def remove_echo_batch(
    rows: pd.DataFrame,
    stores: dict,
    extractor_model: str,
) -> dict:
    """Pre-compute cleaned answers for both-echo pairs via batched VLLMClient calls.

    Creates a single VLLMClient, deduplicates (question_id, model_id) pairs,
    and runs all extractions in one batch call.

    Args:
        rows: DataFrame rows where echo_a=1 and echo_b=1.
        stores: {model_id: ModelAnswerStore}
        extractor_model: vLLM model ID for remove_echo() calls.

    Returns:
        Dict {(question_id, model_id): clean_answer} for all processed answers.
    """
    import json as _json
    from rank_no_eval.client import VLLMClient

    # Deduplicate (question_id, model_id) pairs
    pairs_needed: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        pairs_needed.add((row["question_id"], row["model_a"]))
        pairs_needed.add((row["question_id"], row["model_b"]))

    # Resolve to (pair, sample_dict) — skip missing answers
    valid: list[tuple[tuple[str, str], dict]] = []
    for qid, mid in sorted(pairs_needed):
        try:
            ma = stores[mid].get(qid)
            valid.append(((qid, mid), {"question": ma.question, "answer": ma.answer}))
        except KeyError:
            print(f"  [WARN] Missing answer for ({qid}, {mid}) — skipping.")

    if not valid:
        return {}

    print(f"  Removing echo from {len(valid)} (question_id, model) pairs "
          f"via VLLMClient batch call...")
    client = VLLMClient(extractor_model, EchoExtractionQuery())
    responses = client.collect_model_answers([s for _, s in valid])

    cleaned: dict[tuple[str, str], str] = {}
    for (pair, _), resp in zip(valid, responses):
        try:
            cleaned[pair] = _json.loads(resp)["clean_answer"]
        except (KeyError, _json.JSONDecodeError) as e:
            print(f"  [WARN] remove_echo parse failed for {pair}: {e}")

    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    extractor_model = args.echo_extractor or args.judge

    print(f"Dataset:         {args.dataset}")
    print(f"Judge:           {args.judge}")
    print(f"Judge client:    {args.client}")
    print(f"Intervention:    {args.intervention}")
    print(f"Echo extractor:  {extractor_model}")
    print(f"N:               {args.n}")
    print(f"Seed:            {args.seed}")

    # ------------------------------------------------------------------
    # 1. Sample selection
    # ------------------------------------------------------------------
    pool   = load_sample_pool(args.dataset, args.judge)
    pool   = filter_pool_by_intervention(pool, args.intervention)
    sample = sample_echo_zero(pool, args.n, rng)

    # ------------------------------------------------------------------
    # 2. Load answer texts
    # ------------------------------------------------------------------
    from rank_no_eval.utils import Datasets
    dataset_enum = Datasets(args.dataset)

    all_model_ids = sorted(set(sample["model_a"]) | set(sample["model_b"]))
    stores = load_answer_stores(all_model_ids, dataset_enum)

    # ------------------------------------------------------------------
    # 3. Remove-echo preprocessing for both-echo pairs
    # ------------------------------------------------------------------
    both_echo_rows = sample[(sample["echo_a"] == 1) & (sample["echo_b"] == 1)]
    cleaned: dict[tuple[str, str], str] = {}
    if len(both_echo_rows) > 0:
        print(f"\nPreprocessing {len(both_echo_rows)} both-echo pairs with remove_echo()...")
        cleaned = remove_echo_batch(both_echo_rows, stores, extractor_model)

    # ------------------------------------------------------------------
    # 4. Build intervention records (judge-independent)
    # ------------------------------------------------------------------
    print("\nBuilding intervention records...")
    all_records: list[dict] = []
    n_skipped = 0

    for _, row in sample.iterrows():
        qid     = row["question_id"]
        model_a = row["model_a"]
        model_b = row["model_b"]
        echo_a  = int(row["echo_a"])
        echo_b  = int(row["echo_b"])

        try:
            ma = stores[model_a].get(qid)
            mb = stores[model_b].get(qid)
        except KeyError:
            n_skipped += 1
            continue

        # For both-echo pairs, require pre-cleaned answers
        a_clean = cleaned.get((qid, model_a)) if echo_a == 1 else None
        b_clean = cleaned.get((qid, model_b)) if echo_b == 1 else None
        if echo_a == 1 and (a_clean is None or b_clean is None):
            n_skipped += 1
            continue

        records = build_interventions(
            question_id=qid,
            model_a=model_a,
            model_b=model_b,
            question=ma.question,
            answer_a=ma.answer,
            answer_b=mb.answer,
            echo_a=echo_a,
            echo_b=echo_b,
            starting_echo=echo_a,
            one_correct=int(row["one_correct"]),
            original_preference=int(row["original_preference"]),
            answer_a_clean=a_clean,
            answer_b_clean=b_clean,
        )
        all_records.extend(records)

    print(f"Built {len(all_records)} intervention records  "
          f"({n_skipped} samples skipped due to missing answers)")

    # ------------------------------------------------------------------
    # 5. Run judge on counterfactuals; original preferences already populated
    # ------------------------------------------------------------------
    judged = run_judge(all_records, args.judge, client_type=args.client)

    results_dir = get_results_dir(HERE / "results", args.dataset, args.judge) / args.intervention
    save_results(judged, results_dir / "intervention_raw.csv")

    # ------------------------------------------------------------------
    # 6. Causal analysis
    # ------------------------------------------------------------------
    print("\nRunning causal analysis...")
    run_analysis({args.judge: results_dir})

    print("\nDone.")


if __name__ == "__main__":
    main()
