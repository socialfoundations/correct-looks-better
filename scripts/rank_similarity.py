# Kendall's Tau similarity between different rankings of models
import numpy as np
import os
import json
from pathlib import Path

from utils import (
    data_configs,
    load_accuracy_ranking,
    load_llm_judge_ranking,
    load_model_rankings,
    rank_similarity,
)
from config import BOOTSTRAP_DIR
from rank_no_eval.utils import Datasets

metric_name = {
    "tau": "Kendall's Tau",
    "rho": "Spearman's Rho",
    "r": "Pearson's R",
}

bootstrap_config = {
    "reps": 100,
    "n_samples": {
        "mmlu_pro": 25000,
        "gpqa_diamond": 4000,
        "mt_bench": 1000,
        "simple_qa": 8000,
        "arena_hard": 20000,
        "gsm8k": 50000,
        "gsm8k-ttt": 50000,
        "bbh": 50000,
    },
}

judge_pretty_names = {
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/o3": "o3",
    "microsoft/phi-4": "phi-4",
    "google/gemma-3-27b-it": "gemma-3-27b"
}

def get_method_filename(method: str, judge_model: str, bias: str | None = None, answers_only: bool = False):
    file_name = method
    
    if judge_model:
        judge_model_str = judge_pretty_names.get(judge_model, judge_model)
        file_name += f"_{judge_model_str}"
    if bias:
        if bias == "both":
            bias_str = "style-self-preference"
        else:
            bias_str = bias
        file_name += f"_{bias_str}"
    if answers_only:
        file_name += "_answers-only"

    return file_name + ".npy"


def print_rankings(rank, gt_rank):
    # shuffle rank based on gt order
    rank = sorted(rank, key=lambda x: [m for m, _ in gt_rank].index(x[0]))

    print(f"{'Rank':<5} {'Model ID':<30} {'Accuracy':<15} {'Score':<15}")
    print("-" * 65)

    sorted_gt_rank = sorted(gt_rank, key=lambda x: x[1], reverse=True)
    rank_dict = dict(rank)
    rank_scores_sorted = sorted([s for _, s in rank], reverse=True)

    for i, (model_id, gt_score) in enumerate(sorted_gt_rank):
        score = rank_dict.get(model_id)
        r = rank_scores_sorted.index(score)
        if r == i:
            # matching rank
            print(
                f"{str(i+1)+'.':<5} {model_id[:30]:<30} {gt_score:<15.4f} {score:<15.4f}"
            )
        else:
            # highlight mismatch
            RED = "\033[31m"
            RESET = "\033[0m"
            d = i - r
            direction = "▲" if d > 0 else "▼"
            formatted_score = f"{score:.4f} {RED}({direction} {d}){RESET}"
            print(
                f"{str(i+1)+'.':<5} {model_id[:30]:<30} {gt_score:<15.4f} {formatted_score:<15}"
            )
    print("-" * 65)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mt_bench",
        choices=[d.value for d in Datasets],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rho",
        choices=["tau", "rho", "r"],
        help="Metric to use: Kendall's Tau ('tau'), Spearman's Rho ('rho') or Pearson's R ('r').",
    )
    parser.add_argument(
        "--logit-accuracy",
        action="store_true",
        help="Apply logit transformation to accuracy scores before computing similarity.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BradleyTerry",
        choices=["BradleyTerry", "TrueSkill-M", "WinRate", "AM"],
        help="Ranking method to evaluate. 'AM' = LLM-as-a-judge without ground truth.",
    )
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--judge",
        type=str,
        default="openai/o3",
        choices=list(judge_pretty_names.keys()),
        help="Judge model identifier to use for pairwise ranking.",
    )
    parser.add_argument(
        "--bias",
        type=str,
        default=None,
        choices=["style", "self-preference", "both"],
        help="Bias correction to apply for BradleyTerry: style, self-preference, or both.",
    )
    args = parser.parse_args()
    return args


def load_score_based_ranking(
    dataset: str,
    method: str,
    answers_only: bool,
    bootstrap: bool,
    judge_model: str,
    bias: str | None = None,
):
    if bootstrap:
        n = bootstrap_config["n_samples"][dataset]
        rank = load_model_rankings(
            dataset=dataset,
            method=method,
            n=n,
            judge_model=judge_model,
            shuffle=True,  # bootstrap requires shuffling
            with_replacement=True,  # bootstrap requires sampling with replacement
            answers_only=answers_only,
            bias=bias,
        )
    else:
        rank = load_model_rankings(
            dataset=dataset,
            method=method,
            judge_model=judge_model,
            shuffle=False,
            with_replacement=False,
            answers_only=answers_only,
            bias=bias,
        )
    return rank


def compute_similarity_from_ranking(ground_truth_rank, rank, metric: str, logit_accuracy: bool):
    # sort both rankings by model id for consistency
    ground_truth_rank = sorted(ground_truth_rank, key=lambda x: x[0])
    rank = sorted(rank, key=lambda x: x[0])

    # unpack score based ranks and gt based ranks
    _, model_scores = zip(*rank)
    _, gt_scores = zip(*ground_truth_rank)

    if logit_accuracy:
        # apply logit to accuracy
        gt_scores = np.log(np.array(gt_scores) / (1 - np.array(gt_scores)))

    # compute similarity
    similarity, p_value = rank_similarity(model_scores, gt_scores, method=metric)
    return similarity, p_value


def compute_ranking_similarity(
    dataset: str,
    method: str,
    metric: str,
    bootstrap=False,
    answers_only=False,
    logit_accuracy=False,
    judge_model: str = "openai/gpt-oss-120b",
    bias: str | None = None,
):
    # load ground truth based ranking
    ground_truth_rank = load_accuracy_ranking(dataset) 

    # load score based ranking
    rank = load_score_based_ranking(dataset, method, answers_only, bootstrap, judge_model, bias)

    # print rankings
    print_rankings(rank, ground_truth_rank)

    similarity, p_value = compute_similarity_from_ranking(
        ground_truth_rank=ground_truth_rank,
        rank=rank,
        metric=metric,
        logit_accuracy=logit_accuracy,
    )
    print(
        f"{metric_name[metric]} between {method} and {data_configs[dataset]['ground_truth']}: {similarity:.4f} (p-value: {p_value:.4f})"
    )

    return similarity, p_value


def check_config(base_dir: Path, dataset: str):
    config_path = base_dir / "bootstrap_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            existing = json.load(f)
        if existing["n_samples"][dataset] != bootstrap_config["n_samples"][dataset]:
            raise ValueError(
                f"Bootstrap config mismatch at {config_path}. "
                f"Existing: {existing}, current: {bootstrap_config}"
            )
    else:
        with open(config_path, "w") as f:
            json.dump(bootstrap_config, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    method = args.method

    # BASE_DIR = Path(__file__).resolve().parent / "ranks"
    BASE_DIR = BOOTSTRAP_DIR / f"{args.dataset}" / f"{data_configs[args.dataset]['models_id']}"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Results will be saved to {BASE_DIR}")

    if not args.bootstrap:
        if method == "AM":
            ground_truth_rank = load_accuracy_ranking(dataset)
            am_rank = load_llm_judge_ranking(dataset, judge_model=args.judge)
            print_rankings(am_rank, ground_truth_rank)
            similarity, p_value = compute_similarity_from_ranking(
                ground_truth_rank=ground_truth_rank,
                rank=am_rank,
                metric=args.metric,
                logit_accuracy=args.logit_accuracy,
            )
            print(
                f"{metric_name[args.metric]} between AM w/o GT and {data_configs[dataset]['ground_truth']}: {similarity:.4f} (p-value: {p_value:.4f})"
            )
        else:
            compute_ranking_similarity(
                dataset=dataset,
                method=method,
                metric=args.metric,
                bootstrap=False,
                answers_only=args.answers_only,
                logit_accuracy=args.logit_accuracy,
                judge_model=args.judge,
                bias=args.bias,
            )
    else:
        check_config(BASE_DIR, dataset) # check if config state matches

        metrics_dict = {method: []}
        ranks_dict = {method: []}
        ground_truth_rank = load_accuracy_ranking(dataset)

        n_required = bootstrap_config["reps"]
        file_name = get_method_filename(
            method,
            judge_model=args.judge,
            bias=args.bias,
            answers_only=args.answers_only,
        )
        if not args.rerun and os.path.exists(BASE_DIR / file_name):
            ranks_dict[method] = np.load(
                BASE_DIR / file_name, allow_pickle=True
            ).tolist()[:n_required]
            print(
                f"Loaded {len(ranks_dict[method])} bootstrap rankings from {file_name}"
            )
        n_loaded = len(ranks_dict[method])
        if n_required > n_loaded:
            n_to_generate = n_required - n_loaded
            print(
                f"Generating {n_to_generate} additional bootstrap samples for {method}"
            )
            for _ in range(n_to_generate):
                rank = load_score_based_ranking(
                    dataset=dataset,
                    method=method,
                    answers_only=args.answers_only,
                    bootstrap=True,
                    judge_model=args.judge,
                    bias=args.bias,
                )
                ranks_dict[method].append(rank)
            np.save(BASE_DIR / file_name, np.array(ranks_dict[method], dtype=object))
            print(f"Updated bootstrap rankings saved to {file_name}")
        for rank in ranks_dict[method]:
            similarity, _ = compute_similarity_from_ranking(
                ground_truth_rank=ground_truth_rank,
                rank=rank,
                metric=args.metric,
                logit_accuracy=args.logit_accuracy,
            )
            metrics_dict[method].append(similarity)

        means, cis = [], []
        for method, similarity in metrics_dict.items():
            # Calculate and print statistics
            similarity = np.array(similarity)
            m_sim = np.mean(similarity)
            std_sim = np.std(similarity)
            print(
                f"{metric_name[args.metric]} between {method} and {data_configs[dataset]['ground_truth']}: {m_sim:.4f} +- {std_sim:.4f} (N={len(similarity)})"
            )

            # calculate 95% CI
            ci = 1.96 * std_sim / np.sqrt(len(similarity))
            cis.append(ci)
            means.append(m_sim)

        # print latex style table row
        row = " & ".join([f"{m:.4f} $\\pm$ {ci:.4f}" for m, ci in zip(means, cis)])
        print(f" & {row}")
