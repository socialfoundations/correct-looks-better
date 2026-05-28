import abc
from typing import List
from datasets import Dataset

from rank_no_eval.query.base import BaseQueryMethod


class BaseClient(abc.ABC):
    def __init__(self, model_id, query_method: BaseQueryMethod):
        self.model_id = model_id
        self.query_method = query_method

    @abc.abstractmethod
    def collect_model_answers(self, samples: Dataset, **kwargs) -> List[str]:
        pass
