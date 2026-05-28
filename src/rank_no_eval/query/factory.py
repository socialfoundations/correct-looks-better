from rank_no_eval.utils import Datasets, QueryMode, Task

from .base import BaseQueryMethod
from .MMLUProQueries import MMLUProQueries
from .GPQADiamondQueries import GPQADiamondQueries
from .SimpleQAQueries import SimpleQAQueries
from .GSM8KQueries import GSM8KQueries
from .BBHQueries import BBHQueries


def get_dataset_query_method(
    dataset: Datasets,
    query_mode: QueryMode,
    sycophant: bool = False,
    task: Task = None,
) -> BaseQueryMethod:
    """Return the appropriate query method for the given dataset."""
    if dataset == Datasets.MMLU_PRO:
        return MMLUProQueries(query_mode=query_mode, sycophant=sycophant)
    elif dataset == Datasets.GPQA_DIAMOND:
        return GPQADiamondQueries(query_mode=query_mode, sycophant=sycophant)
    elif dataset == Datasets.SIMPLE_QA:
        return SimpleQAQueries(query_mode=query_mode, sycophant=sycophant)
    elif dataset == Datasets.GSM8K:
        return GSM8KQueries(query_mode=query_mode, sycophant=sycophant)
    elif dataset == Datasets.BBH:
        return BBHQueries(query_mode=query_mode, sycophant=sycophant, task=task)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
