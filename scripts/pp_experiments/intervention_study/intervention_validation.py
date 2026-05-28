"""
Interactive validation of echo interventions.

Samples N answers per intervention type from answer_metadata.csv, applies the
intervention, prompts the user to judge success, and saves results + rates.

  add_echo    — samples answers with is_echo=0, applies add_echo
  remove_echo — samples answers with is_echo=1, applies remove_echo via VLLMClient

Usage (from repo root)::

    python scripts/pp_experiments/intervention_study/intervention_validation.py
    python scripts/pp_experiments/intervention_study/intervention_validation.py --n 5
    python scripts/pp_experiments/intervention_study/intervention_validation.py \\
        --operation remove_echo --extractor-model openai/gpt-oss-20b --n 10

Outputs (written next to answer_metadata.csv)::

    dataset/bbh/intervention_validation.csv
    dataset/bbh/intervention_validation_summary.txt
"""

import argparse
import json as _json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_dataset_dir, load_env

load_env()
from config import DEFAULT_DATASET, DEFAULT_SEED
from interventions import add_echo, EchoExtractionQuery

HERE = Path(__file__).parent

DISPLAY_CHARS = 600
N_ECHO_REPS   = 3   # fixed reps for add_echo during validation


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    half = n // 2
    return text[:half] + f"\n  [...{len(text) - n} chars omitted...]\n  " + text[-half:]


def display_sample(
    i: int,
    total: int,
    operation: str,
    question_id: str,
    model_id: str,
    question: str,
    original: str,
    modified: str,
) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Sample {i}/{total}  |  operation={operation}  |  question_id={question_id}")
    print(f"  model: {model_id}")
    print(sep)
    print(f"QUESTION:\n  {_truncate(question, DISPLAY_CHARS)}")
    print(f"\nORIGINAL ({len(original)} chars):\n  {_truncate(original, DISPLAY_CHARS)}")
    print(f"\nMODIFIED ({len(modified)} chars):\n  {_truncate(modified, DISPLAY_CHARS)}")


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------

HINTS = {
    "add_echo":    "Does the modified answer contain the appended Q+A repetition(s)?",
    "remove_echo": "Was the echo successfully removed, leaving only the final answer?",
}

def prompt_user(operation: str) -> str:
    while True:
        resp = input(
            f"\n  {HINTS[operation]}\n"
            "  [y] yes  [n] no  [s] skip  [q] quit session: "
        ).strip().lower()
        if resp in ("y", "n", "s", "q"):
            return resp
        print("  Please enter y, n, s, or q.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively validate echo interventions."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--operation", choices=["add_echo", "remove_echo", "all"],
                        default="all", help="Which intervention to validate (default: all)")
    parser.add_argument("--n",    type=int, default=10,
                        help="Samples per intervention type (default: 10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--extractor-model", default=None,
        help="vLLM model ID for remove_echo() calls (required when operation includes remove_echo)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    dataset_dir = get_dataset_dir(args.dataset)
    meta_path   = dataset_dir / "answer_metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"answer_metadata.csv not found at {meta_path}.\n"
            f"Run build_dataset.py --dataset {args.dataset} first."
        )

    meta = pd.read_csv(meta_path, dtype={"question_id": str})

    from rank_no_eval.utils import Datasets, QueryMode
    from rank_no_eval.io import load_freeform_dataset
    from rank_no_eval.model_answer import ModelAnswerStore

    dataset_enum = Datasets(args.dataset)
    preloaded    = load_freeform_dataset(dataset=dataset_enum)

    all_operations = [
        ("add_echo",    meta[meta["is_echo"] == 0]),
        ("remove_echo", meta[meta["is_echo"] == 1]),
    ]
    operations = [(op, pool) for op, pool in all_operations
                  if args.operation in (op, "all")]

    validation_rows = []
    parse_errors: dict[str, tuple[int, int]] = {}  # operation → (n_failed, n_attempted)

    for operation, pool in operations:
        if pool.empty:
            print(f"\n[{operation}] No matching answers found — skipping.")
            continue

        n      = min(args.n, len(pool))
        sample = pool.sample(n=n, random_state=int(rng.integers(1 << 31)))

        # Load answer stores only for the models in this sample
        model_ids = sorted(sample["model_id"].unique())
        stores = {
            mid: ModelAnswerStore(mid, preloaded, dataset_enum, QueryMode.FREEFORM)
            for mid in model_ids
        }

        print(f"\n{'='*70}")
        print(f"Validating: {operation}  ({n} samples)")
        print(f"{'='*70}")

        if operation == "add_echo":
            # Compute modifications inline (no LLM needed)
            modifications: dict[tuple[str, str], str] = {}
            for _, row in sample.iterrows():
                try:
                    ma = stores[row["model_id"]].get(row["question_id"])
                    modifications[(row["question_id"], row["model_id"])] = add_echo(
                        ma.question, ma.answer, N_ECHO_REPS
                    )
                except KeyError:
                    pass

        else:
            # remove_echo: pre-compute all modifications in one VLLMClient batch
            if args.extractor_model is None:
                raise ValueError(
                    "--extractor-model is required for remove_echo validation.\n"
                    "Example: --extractor-model openai/gpt-oss-20b"
                )
            from rank_no_eval.client import VLLMClient

            # Collect samples
            batch_pairs: list[tuple[str, str]] = []
            batch_samples: list[dict] = []
            for _, row in sample.iterrows():
                try:
                    ma = stores[row["model_id"]].get(row["question_id"])
                    batch_pairs.append((row["question_id"], row["model_id"]))
                    batch_samples.append({"question": ma.question, "answer": ma.answer})
                except KeyError:
                    pass

            print(f"  Running remove_echo on {len(batch_samples)} samples via VLLMClient...")
            client = VLLMClient(args.extractor_model, EchoExtractionQuery())
            responses = client.collect_model_answers(batch_samples)

            modifications = {}
            n_parse_fail = 0
            for (qid, mid), resp in zip(batch_pairs, responses):
                try:
                    modifications[(qid, mid)] = _json.loads(resp)["clean_answer"]
                except (KeyError, _json.JSONDecodeError) as e:
                    print(f"  [WARN] remove_echo parse failed for ({qid}, {mid}): {e}")
                    n_parse_fail += 1
            parse_errors[operation] = (n_parse_fail, len(batch_pairs))

        # Interactive review loop
        quit_session = False
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            key = (row["question_id"], row["model_id"])
            try:
                ma       = stores[row["model_id"]].get(row["question_id"])
                original = ma.answer
                question = ma.question
            except KeyError as e:
                print(f"\n  [SKIP] Missing answer: {e}")
                continue

            modified = modifications.get(key)
            if modified is None:
                print(f"\n  [SKIP] No modified answer for {key}")
                continue

            display_sample(i, n, operation, row["question_id"], row["model_id"],
                           question, original, modified)
            resp = prompt_user(operation)

            if resp == "q":
                quit_session = True
                break
            if resp == "s":
                continue

            validation_rows.append({
                "operation":       operation,
                "question_id":     row["question_id"],
                "model_id":        row["model_id"],
                "question":        question,
                "original_answer": original,
                "modified_answer": modified,
                "success":         resp == "y",
            })

        if quit_session:
            print("\nSession ended early.")
            break

    if not validation_rows:
        print("\nNo validation results recorded.")
        return

    val_df = pd.DataFrame(validation_rows)

    for op, grp in val_df.groupby("operation"):
        out_dir = HERE / "results" / "interventions" / op
        out_dir.mkdir(parents=True, exist_ok=True)

        samples_cols = ["question_id", "model_id", "question",
                        "original_answer", "modified_answer", "success"]
        grp[samples_cols].to_csv(out_dir / "validation_samples.csv", index=False)
        print(f"\nSaved samples → {out_dir / 'validation_samples.csv'}")

        rate = grp["success"].mean()
        summary_lines = [
            f"Intervention: {op}",
            "=" * 40,
            f"Success rate: {grp['success'].sum()}/{len(grp)}  ({rate:.0%})",
        ]
        if op in parse_errors:
            n_fail, n_total = parse_errors[op]
            summary_lines.append(
                f"Parse error rate: {n_fail}/{n_total}  ({n_fail/n_total:.0%})"
                if n_total else "Parse error rate: N/A (0 attempts)"
            )
        summary = "\n".join(summary_lines)
        print(summary)

        (out_dir / "validation_summary.txt").write_text(summary + "\n")
        print(f"Saved summary → {out_dir / 'validation_summary.txt'}")


if __name__ == "__main__":
    main()
