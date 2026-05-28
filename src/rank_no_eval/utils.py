from dotenv import load_dotenv
import logging
import torch
import random
import os
import numpy as np
import transformers
from enum import Enum, EnumMeta
from typing import List
from dataclasses import dataclass, field, InitVar
from pathlib import Path


# load environment variables
load_dotenv(".env")

# Set up root logger
logger = logging.getLogger("ranking")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()

formatter = logging.Formatter(
    "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(stream_handler)
else:
    logger.handlers.clear()
    logger.addHandler(stream_handler)
# Prevent messages from propagating to the root logger, which may have its own
# handlers added by third-party libraries (e.g. transformers), causing duplicates.
logger.propagate = False

# set up np print options
np.set_printoptions(precision=4, suppress=True)

# check gpu availability
if torch.cuda.is_available():
    specs = torch.cuda.get_device_properties(0)
    logger.info(
        f"Using GPU: {specs.name}, Memory: {specs.total_memory / (1024 ** 3):.2f} GB"
    )

# # Set random seed for reproducibility
REPRODUCIBLE_RANKING = int(os.getenv("REPRODUCIBLE_RANKING", "0"))
SEED = int(os.getenv("SEED", "0"))
if REPRODUCIBLE_RANKING == 1:
    logger.info("Reproducible ranking is enabled. Setting seeds.")
    random.seed(SEED)  # Set a fixed seed for reproducibility
    np.random.seed(SEED)  # Set the seed for NumPy
    torch.manual_seed(SEED)  # Set the seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    transformers.set_seed(
        SEED, deterministic=True
    )  # Set the seed for Hugging Face Transformers
else:
    logger.info("Reproducible ranking is disabled.")


# enums
class QueryMode(Enum):
    MCQ = "MCQ"
    FREEFORM = "freeform"
    PAIRWISE = "pairwise_comparison"
    AM = "answer_matching"


class MatchType(Enum):
    EXACT = "exact_answer_matches"
    PAIRWISE = "pairwise_comparisons"
    AM = "answer_matching"
    AM_NO_GT = "answer_matching_no_gt"


class ClientType(Enum):
    VLLM = "vllm"
    LITELLM = "litellm"


class TaskType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    BINARY = "binary"
    OPEN_ENDED = "open_ended"


# task dataclass
@dataclass(frozen=True)
class Task:
    task_name: str
    short_name: str
    task_type: TaskType
    binary_options: List[str] = field(
        default_factory=list
    )  # mandatory for binary tasks
    versions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BaseDataset:
    dataset_name: str
    partial: bool = False
    ttt: bool = False  # test-time trained models
    tasks: List[Task] = None  # for multitask datasets

    @property
    def is_multitask(self) -> bool:
        return self.tasks is not None and len(self.tasks) > 1

    # task_type specifies the task type for automatic task creation if tasks is None
    task_type: InitVar[TaskType] = None

    def __post_init__(self, task_type: TaskType):
        if self.tasks is None:
            # create a task with the dataset name
            object.__setattr__(
                self, "tasks", [Task(self.dataset_name, "", task_type=task_type)]
            )  # use setattr because of frozen=True

    @property
    def name(self):
        return self.dataset_name + ("-ttt" if self.ttt else "")

    def get_task(self, task_name: str) -> Task:
        for task in self.tasks:
            if task.task_name == task_name or task.short_name == task_name:
                return task
        raise ValueError(f"Task {task_name} not found in dataset {self.dataset_name}.")


class DatasetEnumMeta(EnumMeta):
    def __call__(cls, value, *args, **kwargs):
        # Allow calling with lowercase names
        if isinstance(value, str):
            key = value.upper()
            if key in cls.__members__:
                return cls.__members__[key]
        # Fallback to default behavior
        return super().__call__(value, *args, **kwargs)

    def names(cls):
        # returns a list of all dataset names
        return list(cls.__members__.keys())


class Datasets(Enum, metaclass=DatasetEnumMeta):
    def __new__(cls, dataset):
        obj = object.__new__(cls)
        # .value will be the name property of the dataset
        obj._value_ = dataset.name
        # save dataset object
        obj.obj = dataset
        return obj

    MMLU_PRO = BaseDataset("mmlu_pro", partial=True, task_type=TaskType.MULTIPLE_CHOICE)
    GPQA_DIAMOND = BaseDataset(
        "gpqa_diamond", partial=True, task_type=TaskType.MULTIPLE_CHOICE
    )
    MT_BENCH = BaseDataset("mt_bench", partial=False, task_type=TaskType.OPEN_ENDED)
    SIMPLE_QA = BaseDataset("simple_qa", partial=True, task_type=TaskType.OPEN_ENDED)
    ARENA_HARD = BaseDataset("arena_hard", partial=False, task_type=TaskType.OPEN_ENDED)
    GSM8K = BaseDataset("gsm8k", partial=False, task_type=TaskType.OPEN_ENDED)
    GSM8K_TTT = BaseDataset(
        "gsm8k", partial=False, ttt=True, task_type=TaskType.OPEN_ENDED
    )
    BBH = BaseDataset(
        "bbh",
        partial=False,
        tasks=[  # multiple-choice BBH tasks
            Task("date_understanding", "du", task_type=TaskType.MULTIPLE_CHOICE),
            Task("disambiguation_qa", "dq", task_type=TaskType.MULTIPLE_CHOICE),
            Task("penguins_in_a_table", "pit", task_type=TaskType.MULTIPLE_CHOICE),
            Task("reasoning_about_colored_objects", "rco", task_type=TaskType.MULTIPLE_CHOICE),
            Task(
                "salient_translation_error_detection",
                "sted",
                task_type=TaskType.MULTIPLE_CHOICE,
            ),
            Task("temporal_sequences", "ts", task_type=TaskType.MULTIPLE_CHOICE),
            Task(
                "tracking_shuffled_objects",
                "tso",
                task_type=TaskType.MULTIPLE_CHOICE,
                versions=["three_objects", "five_objects", "seven_objects"],
            ),
        ]
        + [  # binary-choice BBH tasks
            Task(
                "boolean_expressions",
                "be",
                task_type=TaskType.BINARY,
                binary_options=["True", "False"],
            ),
            Task(
                "causal_judgement",
                "cj",
                task_type=TaskType.BINARY,
                binary_options=["Yes", "No"],
            ),
            Task(
                "formal_fallacies",
                "ff",
                task_type=TaskType.BINARY,
                binary_options=["valid", "invalid"],
            ),
            Task("navigate", "nav", task_type=TaskType.BINARY, binary_options=["Yes", "No"]),
            Task(
                "sports_understanding",
                "su",
                task_type=TaskType.BINARY,
                binary_options=["yes", "no"],
            ),
            Task(
                "web_of_lies", "wol", task_type=TaskType.BINARY, binary_options=["Yes", "No"]
            ),
        ]
        + [  # open-ended BBH tasks
            Task("dyck_languages", "dl", task_type=TaskType.OPEN_ENDED),
            Task("multistep_arithmetic_two", "mat", task_type=TaskType.OPEN_ENDED),
            Task("object_counting", "oc", task_type=TaskType.OPEN_ENDED),
            Task("word_sorting", "ws", task_type=TaskType.OPEN_ENDED),
        ],
    )

# helper functions
def model_set_id(models: List[str], n: int = None) -> str:
    import hashlib

    models = [m.replace("/", "--") for m in models]

    sorted_models = sorted(models)  # sort to ensure consistent ordering
    combined = ",".join(sorted_models)
    md5 = hashlib.md5(combined.encode()).hexdigest()
    if n is not None:
        return md5[:n]
    return md5
