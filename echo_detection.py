import argparse
import json
from collections import Counter

from rank_no_eval import EchoDetector
from rank_no_eval.config import ECHO_OUT_DIR
from rank_no_eval.io import load_freeform_dataset
from rank_no_eval.model_answer import ModelAnswerStore
from rank_no_eval.query import get_dataset_query_method
from rank_no_eval.utils import Datasets, QueryMode, logger


def run_model(model_id: str, dataset, query_mode, preloaded_dataset, query_method, detector, debug: bool, override: bool = False) -> None:
    model_slug = model_id.replace("/", "--")
    judge_slug = detector.judge_model.replace("/", "--")
    out_path = ECHO_OUT_DIR / judge_slug / dataset.value / f"{model_slug}.jsonl"
    if out_path.exists() and not override:
        logger.info(f"Skipping {model_id}: results already exist at {out_path}. Use --override to rerun.")
        return

    logger.info(f"Running echo detection for model: {model_id}")
    answer_store = ModelAnswerStore(
        model_id=model_id,
        preloaded_dataset=preloaded_dataset,
        dataset=dataset,
        query_mode=query_mode,
    )

    model_answers = list(answer_store.answer_store.values())
    if debug:
        model_answers = model_answers[:10]
        logger.info("Debug mode: limiting to 10 samples.")
    qa_pairs = [
        f"{query_method.get_prompt({'question': ma.question})}\n{ma.answer}"
        for ma in model_answers
    ]

    # Run echo detection
    results = detector.detect(qa_pairs)

    # Log summary
    counts = Counter(r.value for r in results)
    if debug:
        for ma, qa_pair, result in zip(model_answers, qa_pairs, results):
            logger.debug(f"[{result.value}] {ma.question_id}\n{qa_pair}")
    logger.info(f"Echo detection results for {model_id}: {dict(counts)}")

    if debug:
        logger.info("Debug mode: skipping saving results.")
        return

    # Save results
    output = [
        {"question_id": ma.question_id, "is_echo": result.value}
        for ma, result in zip(model_answers, results)
    ]
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Results saved to {out_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Detect echo failure modes in model responses for a given dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. gsm8k).")
    parser.add_argument("--debug", action="store_true", help="Run on 10 samples only and print each result.")
    parser.add_argument("--override", action="store_true", help="Override existing results if they exist.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model ID to evaluate.")
    model_group.add_argument("--models-file", help="Path to a text file with one model ID per line.")

    args = parser.parse_args()

    if args.models_file:
        with open(args.models_file) as f:
            models = [line.strip() for line in f if line.strip()]
    else:
        models = [args.model]

    dataset = Datasets(args.dataset)
    query_mode = QueryMode.FREEFORM

    preloaded_dataset = load_freeform_dataset(dataset=dataset)
    query_method = get_dataset_query_method(dataset=dataset, query_mode=query_mode)
    detector = EchoDetector()

    for model_id in models:
        run_model(model_id, dataset, query_mode, preloaded_dataset, query_method, detector, args.debug, args.override)


if __name__ == "__main__":
    main()
