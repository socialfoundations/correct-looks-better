"""
Build a dataset for training a pairwise preference classifier.

Usage::
    python build_dataset.py --dataset bbh --judge openai/gpt-oss-20b
    python build_dataset.py --dataset gpqa_diamond --judge openai/gpt4o --models-file models/models-gpqa.txt

Produces two CSV files inside dataset/:
  - answer_metadata.csv: model_id, question_id + flattened style metadata
                        + is_echo + is_correct
                        for every (model, question) pair
  - <judge>/pairwise_preferences.csv : question_id, model_a, model_b, preference
                                (preference=1 if model_a is preferred, 0 if model_b)

"""

from pathlib import Path
import argparse
import json

import pandas as pd

from rank_no_eval.utils import Datasets, QueryMode
from rank_no_eval.io import load_freeform_dataset, load_question_ids
from rank_no_eval.model_answer import ModelAnswerStore
from rank_no_eval.ranker.match_manager import ExactMatchManager, PairwiseMatchManager
from rank_no_eval.config import ECHO_OUT_DIR

from utils import get_dataset_dir, get_dataset_judge_dir

# ---------------------------------------------------------------------------
# Constants (not overridable via CLI)
# ---------------------------------------------------------------------------

ECHO_JUDGE_MODEL = "openai/gpt-oss-120b"
WHAT = "model"  # "model" for TrueSkill-M  /  "answer" for TrueSkill-MA
ANSWERS_ONLY = False

ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"

_DATASET_CHOICES = [d.value for d in Datasets]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise-preference + answer-metadata CSVs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=_DATASET_CHOICES,
        help="Benchmark dataset (e.g. bbh, gpqa_diamond, gsm8k).",
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="Judge model ID whose pairwise preferences to load (e.g. openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help=(
            "Path to a models list file (one model per line). "
            "Defaults to models/models-<dataset>.txt."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested style-metadata dict into a single-level dict.

    Example input::
        {
            "token_len": 120,
            "header_count": {"h1": 0, "h2": 1, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
            "list_count":   {"ordered": 2, "unordered": 5},
            "bold_count":   {"**": 3, "__": 0},
        }
    """
    if metadata is None:
        return {}
    flat: dict = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # replace characters that are awkward in column names
                safe_sub = sub_key.replace("*", "star").replace("_", "und")
                flat[f"{key}__{safe_sub}"] = sub_value
            # also add a total
            flat[f"{key}__total"] = sum(value.values())
        else:
            flat[key] = value
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    dataset: Datasets = Datasets(args.dataset)
    judge_model: str = args.judge

    # Resolve models file
    models_file = (
        Path(args.models_file)
        if args.models_file
        else MODELS_DIR / f"models-{dataset.value}.txt"
    )
    if not models_file.exists():
        raise FileNotFoundError(
            f"Models file not found: {models_file}. "
            f"Provide --models-file explicitly."
        )
    models = [
        line.strip()
        for line in models_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Loaded {len(models)} models from {models_file}")

    # Output directory is keyed by (dataset, judge) so datasets can be shared
    output_dir = get_dataset_judge_dir(dataset.value, judge_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load pairwise preferences
    # ------------------------------------------------------------------
    print(
        f"Loading pairwise preferences for judge '{judge_model}' on {dataset.value}..."
    )
    match_manager = PairwiseMatchManager()
    raw_matches = match_manager.load_matches(
        dataset=dataset,
        model_id=judge_model,
        models=models,
        what=WHAT,
        answers_only=ANSWERS_ONLY,
    )

    if raw_matches is None:
        raise FileNotFoundError(
            f"No pairwise comparison file found for judge='{judge_model}', "
            f"dataset='{dataset.value}', models hash computed from {len(models)} models, "
            f"what='{WHAT}', answers_only={ANSWERS_ONLY}.\n"
            f"Expected file under: {match_manager.matches_base_dir}"
        )

    print(f"  Loaded {len(raw_matches)} raw records.")

    matches_df = (
        pd.DataFrame(raw_matches)
        if not isinstance(raw_matches, pd.DataFrame)
        else raw_matches
    )
    print(f"  Using all {len(matches_df)} records.")

    # Build pairwise_preferences: question_id, model_a, model_b, preference
    # winner == 1  →  model1 (=model_a) is preferred  →  preference = 1
    # winner == 2  →  model2 (=model_b) is preferred  →  preference = 0
    # Ties / unexpected values are dropped with a warning.
    def to_preference(row):
        v = row["winner"]
        try:
            w = int(v)
        except (ValueError, OverflowError):
            return None  # inf, nan, or other non-integer
        if w == 1:
            return 1
        elif w == 2:
            return 0
        return None  # tie or unknown

    matches_df["preference"] = matches_df.apply(to_preference, axis=1)
    n_dropped = matches_df["preference"].isna().sum()
    if n_dropped:
        print(f"  Warning: dropping {n_dropped} rows with unexpected winner values.")
    matches_df = matches_df.dropna(subset=["preference"])
    matches_df["preference"] = matches_df["preference"].astype(int)

    # Rename columns for clarity
    pref_df = matches_df.rename(columns={"model1": "model_a", "model2": "model_b"})
    # Use question1 as the canonical question_id
    # (for WHAT=="model", question1 == question2)
    pref_df = pref_df.rename(columns={"question1": "question_id"})
    pref_df = pref_df[["question_id", "model_a", "model_b", "preference"]]

    pref_path = output_dir / "pairwise_preferences.csv"
    pref_df.to_csv(pref_path, index=False)
    print(f"  Saved pairwise preferences → {pref_path}  ({len(pref_df)} rows)")

    # ------------------------------------------------------------------
    # 2. Build answer metadata (shared across judges, keyed by dataset)
    # ------------------------------------------------------------------
    meta_dir = get_dataset_dir(dataset.value)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "answer_metadata.csv"

    if meta_path.exists():
        print(f"\nAnswer metadata already exists at {meta_path}, skipping rebuild.")
    else:
        print(
            f"\nBuilding answer metadata for all {len(models)} models x all questions..."
        )

        preloaded_dataset = load_freeform_dataset(dataset=dataset)
        all_question_ids = load_question_ids(dataset=dataset)
        print(f"  Dataset contains {len(all_question_ids)} questions.")

        # ------------------------------------------------------------------
        # 2b. Load exact-match correctness labels per model
        # ------------------------------------------------------------------
        print("  Loading exact-match correctness labels...")
        exact_match_manager = ExactMatchManager()

        def load_correctness(model_id: str) -> dict[str, int]:
            """Return {question_id: is_correct} for the given model across all tasks."""
            correct: dict[str, int] = {}
            tasks = getattr(dataset.obj, "tasks", [])
            if tasks:
                for task in tasks:
                    df = exact_match_manager.load_matches(dataset, model_id, task=task)
                    if df is None:
                        continue
                    for _, r in df.iterrows():
                        correct[str(r["question_id"])] = int(
                            str(r["extracted_answer"]).lower()
                            == str(r["answer"]).lower()
                        )
            else:
                df = exact_match_manager.load_matches(dataset, model_id)
                if df is not None:
                    for _, r in df.iterrows():
                        correct[str(r["question_id"])] = int(
                            str(r["extracted_answer"]).lower()
                            == str(r["answer"]).lower()
                        )
            return correct

        metadata_rows = []
        for model_id in models:
            print(f"  Loading answers for {model_id} ...")
            try:
                store = ModelAnswerStore(
                    model_id=model_id,
                    preloaded_dataset=preloaded_dataset,
                    dataset=dataset,
                    query_mode=QueryMode.FREEFORM,
                    with_metadata=True,
                )
            except Exception as exc:
                print(f"    Warning: could not load answers for {model_id}: {exc}")
                continue

            # Load echo detection labels for this model
            echo_judge_safe = ECHO_JUDGE_MODEL.replace("/", "--")
            model_safe = model_id.replace("/", "--")
            echo_path = (
                ECHO_OUT_DIR / echo_judge_safe / dataset.value / f"{model_safe}.jsonl"
            )
            echo_labels: dict[str, int] = {}
            if echo_path.exists():
                with open(echo_path) as f:
                    for line in f:
                        rec = json.loads(line)
                        echo_labels[str(rec["question_id"])] = (
                            1 if rec["is_echo"] == "ECHO" else 0
                        )
            else:
                print(f"    Warning: no echo detection file found at {echo_path}")

            # Load correctness labels for this model
            correctness = load_correctness(model_id)
            if not correctness:
                print(
                    f"    Warning: no exact-match correctness data found for {model_id}"
                )

            for question_id in all_question_ids:
                try:
                    answer = store.get(question_id)
                except KeyError:
                    print(
                        f"    Warning: question_id {question_id} not found for {model_id}"
                    )
                    continue

                row = {
                    "model_id": model_id,
                    "question_id": question_id,
                    **flatten_metadata(answer.metadata),
                    "is_echo": echo_labels.get(question_id),
                    "is_correct": correctness.get(question_id),
                }
                metadata_rows.append(row)

        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(meta_path, index=False)
        print(f"\n  Saved answer metadata → {meta_path}  ({len(meta_df)} rows)")


if __name__ == "__main__":
    main()
