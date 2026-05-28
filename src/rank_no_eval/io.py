import json
from typing import List, Union, Dict, Optional
from datasets import load_dataset, Dataset, Value, concatenate_datasets
from pathlib import Path
import random


from .utils import QueryMode, Datasets, Task
from .utils import logger
from .config import (
    ANSWERS_OUT_DIR,
    ANSWERS_WITH_METADATA_OUT_DIR,
    MMLU_PRO_IDS_PATH,
    GPQA_DIAMOND_IDS_PATH,
    SIMPLE_QA_IDS_PATH,
    GSM8K_IDS_PATH,
    BBH_IDS_PATH,
)


def _answers_file_path(
    model_id: str,
    query_mode: QueryMode,
    dataset: Datasets,
    task: Task = None,
    with_metadata: bool = False,
) -> Path:
    base = ANSWERS_OUT_DIR if not with_metadata else ANSWERS_WITH_METADATA_OUT_DIR
    if task:
        answers_dir = base / query_mode.value / dataset.value / task.task_name
    else:
        answers_dir = base / query_mode.value / dataset.value
    answers_file = answers_dir / f"{model_id.replace('/', '--')}.jsonl"
    # create directories if they do not exist
    answers_file.parent.mkdir(parents=True, exist_ok=True)
    return answers_file


def load_answers(
    model_id: str,
    query_mode: QueryMode,
    dataset: Datasets,
    task: Task = None,
    with_metadata: bool = False,
) -> Union[List[Dict[str, str]], None]:
    answers_file = _answers_file_path(
        model_id, query_mode, dataset, task, with_metadata
    )
    logger.debug(f"Loading model answers from {answers_file}...")
    if answers_file.exists():
        with open(answers_file, "r") as f:
            answers = [json.loads(line) for line in f]
            return answers
    else:
        return None


def save_answers(
    model_id: str,
    answers: List[Dict[str, str]],
    query_mode: QueryMode,
    dataset: Datasets,
    task: Task = None,
):
    sorted_answers = sorted(answers, key=lambda x: x["question_id"])
    answers_file = _answers_file_path(model_id, query_mode, dataset, task)
    with open(answers_file, "w") as f:
        for a in sorted_answers:
            f.write(
                json.dumps({"question_id": a["question_id"], "answer": a["answer"]})
                + "\n"
            )

    logger.debug(f"Model answers saved to {answers_file}")


def _preprocess_mmlu_pro(data: Dataset) -> Dataset:
    logger.debug("Preprocessing MMLU-Pro dataset...")
    return data.cast_column("question_id", Value("string"))


def _load_mmlu_pro_dataset():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    dataset = _preprocess_mmlu_pro(dataset["test"])
    return dataset


def _preprocess_gpqa_diamond(data: Dataset) -> Dataset:
    logger.debug("Preprocessing GPQA Diamond dataset...")

    def add_choices_and_correct_answer(sample: dict) -> dict:
        choices = [
            sample["Correct Answer"],
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]
        # Use a seed based on the Record ID for reproducible shuffling
        random.seed(sample["Record ID"])
        random.shuffle(choices)
        corr_index = choices.index(sample["Correct Answer"])
        corr_choice = chr(65 + corr_index)
        sample["question"] = sample["Question"]
        sample["choices"] = choices
        sample["answer"] = corr_choice
        sample["question_id"] = str(sample["Record ID"])
        return sample

    return data.map(add_choices_and_correct_answer, remove_columns=data.column_names)


def _load_gpqa_diamond_dataset():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    dataset = _preprocess_gpqa_diamond(dataset["train"])
    return dataset


def _preprocess_mt_bench(data: Dataset) -> Dataset:
    logger.debug("Preprocessing MT-Bench dataset...")

    def add_question(sample: dict) -> dict:
        # 0th turn instruction is the question
        sample["question"] = sample["turns"][0]
        return sample

    return data.map(add_question)


def _load_mt_bench_dataset():
    dataset = load_dataset("dim/mt_bench_en")
    dataset = _preprocess_mt_bench(dataset["train"])
    return dataset.cast_column("question_id", Value("string"))


def _preprocess_simple_qa(data: Dataset) -> Dataset:
    logger.debug("Preprocessing SimpleQA dataset...")

    def preprocess(sample: dict, idx) -> dict:
        sample["question_id"] = str(idx)
        sample["question"] = sample["problem"]
        return sample

    return data.map(preprocess, with_indices=True)


def _load_simple_qa_dataset():
    dataset = load_dataset("basicv8vc/SimpleQA")
    dataset = _preprocess_simple_qa(dataset["test"])
    return dataset


def _preprocess_arena_hard(data: Dataset) -> Dataset:
    logger.debug("Preprocessing Arena Hard dataset...")

    def preprocess(sample: dict) -> dict:
        sample["question"] = sample["turns"][0]["content"]
        return sample

    return data.map(preprocess)


def _load_arena_hard_dataset():
    dataset = load_dataset("lmarena-ai/arena-hard-auto-v0.1")
    dataset = _preprocess_arena_hard(dataset["train"])
    return dataset


def _preprocess_gsm8k(data: Dataset) -> Dataset:
    logger.debug("Preprocessing GSM8K dataset...")

    def preprocess(sample: dict, idx) -> dict:
        sample["question_id"] = str(idx)
        sample["question"] = sample["question"]
        return sample

    return data.map(preprocess, with_indices=True)


def _load_gsm8k_dataset():
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = _preprocess_gsm8k(dataset["test"])
    return dataset


def _preprocess_bbh(data: Dataset) -> Dataset:
    logger.debug("Preprocessing BBH dataset...")

    def preprocess(sample: dict, idx) -> dict:
        sample["question_id"] = str(idx)
        sample["question"] = sample["question"]
        choices = sample.get("choices", None)
        if choices:
            sample["choices"] = [
                {"label": label.rstrip(")"), "text": text}
                for label, text in zip(choices["label"], choices["text"])
            ]  # Transform labels: "A)" -> "A"
        sample["answer"] = sample["target"].strip()
        return sample

    return data.map(preprocess, with_indices=True)


def _load_bbh_dataset(task: Optional[Task] = None) -> Dataset:
    if task is not None:
        # Load a specific task
        assert (
            task in Datasets.BBH.obj.tasks
        ), f"Unsupported task '{task.task_name}' for BBH dataset."
        tasks_to_load = [task]
    else:
        # Load all tasks
        tasks_to_load = Datasets.BBH.obj.tasks

    datasets_list = []
    for current_task in tasks_to_load:
        prefix = current_task.short_name
        if current_task.versions:
            version_datasets = []
            for version in current_task.versions:
                dataset_loaded = load_dataset(
                    "Joschka/big_bench_hard", f"{current_task.task_name}_{version}"
                )
                version_datasets.append(
                    _preprocess_bbh(dataset_loaded[current_task.task_name + f"_{version}"])
                )
            dataset_loaded = concatenate_datasets(version_datasets)
            # rebuild question_ids after concatenation
            dataset_loaded = dataset_loaded.map(
                lambda sample, idx: {"question_id": f"{prefix}_{idx}"}, with_indices=True
            )
        else:
            dataset_loaded = load_dataset("Joschka/big_bench_hard", current_task.task_name)
            dataset_loaded = _preprocess_bbh(dataset_loaded[current_task.task_name])
            # add prefix to question_ids
            dataset_loaded = dataset_loaded.map(
                lambda sample, idx: {"question_id": f"{prefix}_{idx}"}, with_indices=True
            )
        datasets_list.append(dataset_loaded)

    # Concatenate all task datasets
    if len(datasets_list) == 1:
        return datasets_list[0]
    else:
        return concatenate_datasets(datasets_list)


def load_freeform_ids(split: str = "mmlu_pro") -> List[str]:
    data = load_dataset("nikhilchandak/freeform-datasets")
    split = data[split]
    return split["question_id"]


def load_full_dataset(dataset: Datasets = Datasets.MMLU_PRO, **kwargs) -> Dataset:
    if dataset == Datasets.MMLU_PRO:
        return _load_mmlu_pro_dataset()
    elif dataset == Datasets.GPQA_DIAMOND:
        return _load_gpqa_diamond_dataset()
    elif dataset == Datasets.MT_BENCH:
        return _load_mt_bench_dataset()
    elif dataset == Datasets.SIMPLE_QA:
        return _load_simple_qa_dataset()
    elif dataset == Datasets.ARENA_HARD:
        return _load_arena_hard_dataset()
    elif dataset in [Datasets.GSM8K, Datasets.GSM8K_TTT]:
        return _load_gsm8k_dataset()
    elif dataset == Datasets.BBH:
        return _load_bbh_dataset(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


dset_to_ids_filepath = {
    Datasets.MMLU_PRO: MMLU_PRO_IDS_PATH,
    Datasets.GPQA_DIAMOND: GPQA_DIAMOND_IDS_PATH,
    Datasets.SIMPLE_QA: SIMPLE_QA_IDS_PATH,
    Datasets.GSM8K: GSM8K_IDS_PATH,
    Datasets.BBH: BBH_IDS_PATH,
}


def load_ids_from_file(dataset: Datasets) -> List[str] | None:
    ids_filepath = dset_to_ids_filepath.get(dataset, None)

    if ids_filepath and ids_filepath.exists():
        logger.debug(f"Loading ids from {ids_filepath}...")
        with open(ids_filepath, "r") as file:
            ids = [str(line.strip()) for line in file if line.strip()]
        return ids
    return None


def save_ids(dataset: Datasets, ids: List[str]):
    ids_filepath = dset_to_ids_filepath.get(dataset, None)
    if ids_filepath:
        logger.debug(f"Saving ids to {ids_filepath}...")
        with open(ids_filepath, "w") as file:
            file.write("\n".join(ids))


def create_partial_ids(dataset: Datasets, **kwargs) -> List[str]:
    if dataset in [Datasets.MMLU_PRO, Datasets.GPQA_DIAMOND]:
        # freeform ids - set of ids where questions can have freeform (open-ended) answers
        logger.debug(
            "Loading freeform question ids from nikhilchandak/freeform-datasets..."
        )
        ids = load_freeform_ids(split=dataset)
    elif dataset == Datasets.SIMPLE_QA:
        # random ids - when all questions can have freeform answers
        data = _load_simple_qa_dataset()
        logger.debug(f"Sampling 500 question ids from dataset (N={len(data)})...")
        ids = list(data.shuffle(seed=42).select(range(500))["question_id"])
    elif dataset == Datasets.BBH:
        ids = []
        for task in Datasets.BBH.obj.tasks:
            logger.debug(f"Loading BBH task '{task.task_name}' dataset...")
            data = _load_bbh_dataset(task=task)
            logger.debug(
                f"Sampling 20% of question ids from BBH task '{task.task_name}' dataset (N={len(data)})..."
            )
            subset_size = int(0.2 * len(data))
            ids.extend(list(data.shuffle(seed=42).select(range(subset_size))["question_id"]))

    return ids


def load_question_ids(dataset: Datasets, **kwargs) -> List[str]:
    ids = load_ids_from_file(dataset)
    if ids is not None:
        return ids
    else:
        ids = create_partial_ids(dataset, **kwargs) if dataset.obj.partial else list(load_full_dataset(dataset, **kwargs)["question_id"])
        save_ids(dataset, ids)
        return ids


def load_partial_dataset(
    dataset: Datasets = Datasets.MMLU_PRO, ids: Optional[List[str]] = None, **kwargs
) -> Dataset:
    if ids is None:
        ids = load_question_ids(dataset, **kwargs)

    # filter for ids
    data = load_full_dataset(dataset, **kwargs)
    test_filtered = data.filter(lambda x: x["question_id"] in ids)

    logger.debug(f"Loaded {len(test_filtered)} samples from {dataset} dataset")

    return test_filtered


def load_freeform_dataset(dataset: Datasets, **kwargs) -> Dataset:
    # load questions from datasets that support freeform answers
    if dataset.obj.partial:
        return load_partial_dataset(dataset, **kwargs)
    return load_full_dataset(dataset, **kwargs)
