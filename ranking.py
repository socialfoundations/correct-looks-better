import argparse
import os
import pandas as pd
from pathlib import Path
from typing import List

from rank_no_eval.config import RANKING_OUTPUT_DIR, OUT_DIR
from rank_no_eval.utils import ClientType, Datasets, model_set_id, logger
from rank_no_eval import (
    ModelAccuracy,
    AccuracyMCQRanker,
    PairwiseComparisonRanker,
    ModelAnswerTrueSkill,
    ModelTrueSkill,
    ModelWinRate,
    ModelElo,
    ModelAnswerMatch,
    AnswerMatchRanker,
    ModelBradleyTerry,
    ModelMultiTaskAccuracy,
    ModelMultiTaskAnswerMatch,
)
from rank_no_eval.io import load_question_ids


def save_model_set_ids(models: List[str], gen_id: str):
    model_set_file = OUT_DIR / "model_set_ids.txt"
    existing_ids = []
    if model_set_file.exists():
        with open(model_set_file, "r") as f:
            for line in f:
                id, _ = line.strip().split(":", 1)
                existing_ids.append(id)
    if gen_id in existing_ids:
        logger.debug(
            f"ID for set of models {gen_id[:8]} already exists. Skipping save."
        )
        return
    with open(model_set_file, "a") as f:
        f.write(f"{gen_id}: {', '.join(models)}\n")


def save_ranking(
    ranking_data,
    columns,
    ranking_output_dir: Path,
    filename: str,
):
    os.makedirs(ranking_output_dir, exist_ok=True)
    pd.DataFrame(ranking_data, columns=columns[: len(ranking_data[0])]).to_csv(
        ranking_output_dir / filename, index=False
    )


def get_eval_objects(args, model_ids):
    if args.score == "Pairwise":
        if args.what == "TrueSkill-M":
            return [ModelTrueSkill(id=mid) for mid in model_ids]
        elif args.what == "WinRate":
            return [ModelWinRate(id=mid) for mid in model_ids]
        elif args.what == "BradleyTerry":
            return [ModelBradleyTerry(id=mid) for mid in model_ids]
        elif args.what == "Elo":
            return [ModelElo(id=mid) for mid in model_ids]
        elif args.what == "TrueSkill-MA":
            question_ids = load_question_ids(Datasets(args.dataset))
            return [
                ModelAnswerTrueSkill(model_id=mid, question_id=q_id)
                for q_id in question_ids
                for mid in model_ids
            ]
    elif args.score == "AM":
        if args.dataset == Datasets.BBH.value:
            tasks = [task.task_name for task in Datasets.BBH.obj.tasks]
            return [ModelMultiTaskAnswerMatch(id=mid, tasks=tasks) for mid in model_ids]
        else:
            return [ModelAnswerMatch(id=mid) for mid in model_ids]
    else:
        if args.dataset == Datasets.BBH.value:
            tasks = [task.task_name for task in Datasets.BBH.obj.tasks]
            return [ModelMultiTaskAccuracy(id=mid, tasks=tasks) for mid in model_ids]
        else:
            return [ModelAccuracy(id=mid) for mid in model_ids]


def create_ranker(args, eval_objects):
    if args.score == "Pairwise":
        return PairwiseComparisonRanker(
            dataset=args.dataset,
            judge_model=args.judge_model,
            client_type=(
                ClientType.LITELLM if args.client == "litellm" else ClientType.VLLM
            ),
            eval_objects=eval_objects,
            answers_only=args.answers_only,
            control_for=args.control_for,
            pair_filter=args.pair_filter,
        )
    elif args.score == "AM":
        return AnswerMatchRanker(
            dataset=args.dataset,
            judge_model=args.judge_model,
            client_type=(
                ClientType.LITELLM if args.client == "litellm" else ClientType.VLLM
            ),
            eval_objects=eval_objects,
            with_ground_truth=args.with_ground_truth,
        )
    else:
        return AccuracyMCQRanker(
            dataset=args.dataset,
            eval_objects=eval_objects,
        )


def process_ranking(args, ranking, gen_id):
    ranking_output_dir = Path(
        RANKING_OUTPUT_DIR,
        args.dataset,
        gen_id[:8],
    )
    if args.score == "Pairwise":
        if args.rank == "model":
            model_ranking = ranking.model_ranking()
            logger.info(f"Model ranking: {model_ranking}")
            filename = f"{args.what}"
            if args.control_for != []:
                for control in sorted(args.control_for):
                    filename += f"_{control}"
                filename += "_control"
            if args.judge_model == "human":
                filename += "-human"
            save_ranking(
                ranking_data=model_ranking,
                columns=["model_id", "score", "uncertainty"],
                ranking_output_dir=ranking_output_dir,
                filename=f"{filename}.csv",
            )
        elif args.rank == "question":
            question_ranking = ranking.question_ranking()
            logger.info(f"Question ranking: {question_ranking}")
            save_ranking(
                ranking_data=question_ranking,
                columns=["question_id", "score", "uncertainty"],
                ranking_output_dir=ranking_output_dir,
                filename="QuestionRanking.csv",
            )
    elif args.score == "AM" and not args.with_ground_truth:
        model_ranking = ranking.model_ranking()
        logger.info(f"Model ranking: {model_ranking}")
        save_ranking(
            ranking_data=model_ranking,
            columns=["model_id", "score", "uncertainty"],
            ranking_output_dir=ranking_output_dir,
            filename=f"LLM-as-a-judge-noGT.csv",
        )
    else:
        model_ranking = ranking.model_ranking()
        logger.info(f"Model ranking: {model_ranking}")
        save_ranking(
            ranking_data=model_ranking,
            columns=["model_id", "score"],
            ranking_output_dir=ranking_output_dir,
            filename="Accuracy.csv",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base - Accuracy based ranking
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models-file",
        type=str,
        help="File with model IDs to rank.",
    )
    group.add_argument(
        "--model-name",
        type=str,
        help="Model to rank.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu_pro",
        choices=[ds.value for ds in Datasets],
    )
    subparsers = parser.add_subparsers(dest="score")

    # Answer Matching Judge based ranking
    am_parser = subparsers.add_parser("AM")
    am_parser.add_argument(
        "--judge-model", type=str, default="openai/o3"
    )
    am_parser.add_argument(
        "--client", type=str, default="litellm", choices=["vllm", "litellm"]
    )
    am_parser.add_argument(
        "--run-async", action="store_true", help="Run in async mode."
    )
    am_parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Use ground truth answers for answer matching.",
    )

    # Pairwise Judge based ranking
    trueskill_parser = subparsers.add_parser("Pairwise")
    trueskill_parser.add_argument(
        "--judge-model", type=str, default="openai/o3"
    )
    trueskill_parser.add_argument(
        "--client", type=str, default="litellm", choices=["vllm", "litellm"]
    )
    trueskill_parser.add_argument(
        "--what",
        type=str,
        default="TrueSkill-MA",
        choices=["TrueSkill-MA", "TrueSkill-M", "WinRate", "Elo", "BradleyTerry"],
    )
    trueskill_parser.add_argument(
        "-n", type=int, help="Number of pairwise comparisons. Must be >= 0. Pass 0 (or omit) to use the full set of pairs."
    )
    trueskill_parser.add_argument(
        "--run-async", action="store_true", help="Run in async mode."
    )
    trueskill_parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model responses in out/raw/ directory. Works only with Litellm client.",
    )
    trueskill_parser.add_argument(
        "--rank", type=str, default="model", choices=["model", "question"]
    )
    trueskill_parser.add_argument(
        "--load-more", action="store_true", help="Enable load more"
    )
    trueskill_parser.add_argument(
        "--load-additional",
        action="store_true",
        help="Load additional n samples instead of n_saved - n.",
    )
    trueskill_parser.add_argument(
        "--answers-only",
        action="store_true",
        help="Use only answers for ranking. Questions will not be shown to judge model.",
    )
    trueskill_parser.add_argument(
        "--pair-filter",
        type=str,
        default=None,
        choices=["verifiable", "unverifiable"],
        help="Filter pairs by correctness: 'verifiable' (one correct, one incorrect) or 'unverifiable' (both correct or both incorrect).",
    )
    trueskill_parser.add_argument(
        "--control-for",
        nargs="*",
        type=str,
        default=[],
        choices=["style", "model_family", "self-preference"],
        help="Control for variables (choose from ['style', 'model_family', 'self-preference']).",
    )
    args = parser.parse_args()

    if args.models_file:
        if not os.path.exists(args.models_file):
            raise FileNotFoundError(f"Models file {args.models_file} does not exist.")
        with open(args.models_file, "r") as f:
            model_ids = [line.strip() for line in f if line.strip()]
        gen_id = model_set_id(model_ids)
        logger.info(f"List of models: {model_ids} (ID: {gen_id[:8]})")
        save_model_set_ids(model_ids, gen_id)
    elif args.model_name:
        model_ids = [args.model_name]
        logger.info(f"Model: {args.model_name}")

    eval_objects = get_eval_objects(args, model_ids)
    ranking = create_ranker(args, eval_objects)
    if args.score == "Pairwise":
        if args.n is not None and args.n < 0:
            raise ValueError(f"n must be >= 0, got {args.n}")
        n = None if (args.n is None or args.n == 0) else args.n
        ranking.rank(
            n=n,
            run_async=args.run_async,
            save_raw=args.save_raw,
            load_more=args.load_more,
            load_additional=args.load_additional,
        )
    elif args.score == "AM":
        ranking.rank(run_async=args.run_async)
    else:
        ranking.rank()

    # don't save ranking if there is only one model
    if len(eval_objects) > 1:
        # save resulting model / question ranking
        process_ranking(args, ranking, gen_id=gen_id)

        logger.debug(ranking)
