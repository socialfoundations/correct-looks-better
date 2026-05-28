from argparse import ArgumentParser

from rank_no_eval import ModelPrompter
from rank_no_eval.utils import logger, QueryMode, ClientType, Datasets, TaskType


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(
        description="Evaluate MMLU-Pro dataset with a specified model."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu_pro",
        choices=[d.value for d in Datasets],
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="MCQ",
        choices=["MCQ", "freeform"],
        help="Evaluation mode: MCQ or freeform",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun the evaluation even if output files exist",
    )
    parser.add_argument(
        "--litellm",
        action="store_true",
        help="Use litellm instead of vllm",
    )
    parser.add_argument(
        "--run-async",
        action="store_true",
        help="Run litellm requests asynchronously",
    )
    parser.add_argument(
        "--sycophant",
        action="store_true",
        help="Instruct model to give sycophantic responses.",
    )
    args = parser.parse_args()

    logger.info(f"Running MMLU-Pro queries for model: {args.model_name}")
    client_type = ClientType.LITELLM if args.litellm else ClientType.VLLM
    if args.dataset == Datasets.BBH.value:
        query = None
        for task in Datasets.BBH.obj.tasks:
            if (
                args.query_mode == QueryMode.MCQ.value
                and not task.task_type == TaskType.MULTIPLE_CHOICE
            ):
                logger.info(
                    f"Skipping BBH task ({task.task_name}) in MCQ mode as it has no MCQ format."
                )
                continue
            if query is None:
                query = ModelPrompter(
                    model_id=args.model_name,
                    dataset=args.dataset,
                    query_mode=QueryMode(args.query_mode),
                    client_type=client_type,
                    task=task,
                )
            else:
                query.set_task(task)
            logger.info(
                f"Running BBH task ({task.task_name}) in {args.query_mode} mode."
            )
            query.run(
                rerun=args.rerun, run_async=args.run_async, sycophant=args.sycophant
            )
    else:
        query = ModelPrompter(
            model_id=args.model_name,
            dataset=args.dataset,
            query_mode=QueryMode(args.query_mode),
            client_type=client_type,
        )
        query.run(rerun=args.rerun, run_async=args.run_async, sycophant=args.sycophant)
