import argparse
from pathlib import Path

from dotenv import load_dotenv

DATASET_BASE_DIR = Path(__file__).parent / "dataset"
_REPO_ROOT = Path(__file__).parents[2]


def load_env() -> None:
    """Load .env from the repository root."""
    load_dotenv(_REPO_ROOT / ".env")


def get_dataset_dir(dataset: str) -> Path:
    """Return the canonical dataset directory for a dataset name.

    Example: get_dataset_dir("bbh") → pp_experiments/dataset/bbh
    """
    return DATASET_BASE_DIR / dataset


def get_dataset_judge_dir(dataset: str, judge: str) -> Path:
    """Return the canonical dataset directory for a (dataset, judge) pair.

    Example: get_dataset_judge_dir("bbh", "openai/gpt-oss-20b")
             → pp_experiments/dataset/bbh/openai--gpt-oss-20b
    """
    judge_safe = judge.replace("/", "--")
    return DATASET_BASE_DIR / dataset / judge_safe


def get_results_dir(base: Path, dataset: str, judge: str) -> Path:
    """Return *base*/<dataset>__<judge_safe> for a (dataset, judge) pair.

    Example: get_results_dir(HERE / "results", "bbh", "openai/gpt-oss-20b")
             → matching_study/results/bbh__openai--gpt-oss-20b
    """
    judge_safe = judge.replace("/", "--")
    return base / f"{dataset}__{judge_safe}"


def parse_args(
    description: str = "",
    default_dataset: str = "",
    default_judge: str = "",
) -> argparse.Namespace:
    """Shared CLI parser for matching-study scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", default=default_dataset,
        help=f"Dataset name (default: {default_dataset})",
    )
    parser.add_argument(
        "--judge", default=default_judge,
        help=f"Judge model ID (default: {default_judge})",
    )
    return parser.parse_args()
