# base class for ranking models

import abc
import random
from datetime import datetime
from typing import Dict, Tuple, List, Union
from pathlib import Path

from rank_no_eval.utils import Datasets, MatchType, logger
from rank_no_eval.eval_object.base import BaseEvalObject
from rank_no_eval.config import MATCHES_OUT_DIR


class MatchManager(abc.ABC):
    def __init__(self, match_type: MatchType):
        self.match_type = match_type

    @property
    def matches_base_dir(self) -> Path:
        return MATCHES_OUT_DIR / self.match_type.value

    @abc.abstractmethod
    def match_file(self, dataset: Datasets, model_id: str, **kwargs) -> Path:
        pass

    @abc.abstractmethod
    def load_matches(
        self, dataset: Datasets, model_id: str, **kwargs
    ) -> Union[List, None]:
        pass

    @abc.abstractmethod
    def save_matches(self, matches, dataset: Datasets, model_id: str, **kwargs):
        pass


class BaseRanker(abc.ABC):

    def __init__(
        self,
        dataset: str,
        match_manager: MatchManager,
        eval_objects: list[BaseEvalObject] = None,
    ):
        self.match_manager = match_manager
        if eval_objects is None:
            eval_objects = []
        self.eval_objects = eval_objects
        self.dataset = Datasets(dataset)

    @abc.abstractmethod
    def rank(self):
        pass

    @abc.abstractmethod
    def get_matches(self, *args, **kwargs):
        pass

    def add_eval_object(self, eval_object: BaseEvalObject):
        """Add an eval object to the ranker."""
        self.eval_objects.append(eval_object)

    def __repr__(self) -> str:
        text = "Ranking:\n"
        for i, r in enumerate(self.eval_objects, 1):
            text += f"{i}. {r}\n"

        return text

    @abc.abstractmethod
    def model_ranking(self) -> Tuple[str, float]:
        """Return model IDs and their score in descending order."""
        pass

    def sample_items(self, items, n: int, shuffle: bool = False, with_replacement: bool = False):
        if items is None:
            raise ValueError("Cannot sample from None items.")
        if n is None:
            n = int(len(items) * 0.8) # default to 80% of items
            logger.debug(f"Sampling default: sampling {n}/{len(items)} items.")
        
         # handle pandas DataFrame
        try:
            import pandas as pd

            if isinstance(items, pd.DataFrame):
                if shuffle:
                    if not with_replacement:
                        n = min(n, len(items))
                    return items.sample(n=n, replace=with_replacement)
                return items.head(n)
        except Exception:
            pass

        items = list(items)
        if shuffle:
            random.seed(datetime.now())
            if with_replacement:
                return random.choices(items, k=n)
            n = min(n, len(items))
            return random.sample(items, n)
        return items[:n]

    def sample_multitask_questions(
        self, n: int = None, shuffle: bool = False, with_replacement: bool = False
    ) -> Tuple[dict, int]:
        if not self.dataset.obj.is_multitask:
            raise ValueError("sample_multitask_questions called on non-multitask dataset")

        from rank_no_eval.io import load_question_ids

        task_question_pairs = []
        for task in self.dataset.obj.tasks:
            question_ids = load_question_ids(self.dataset, task=task)
            task_question_pairs.extend([(task, q_id) for q_id in question_ids])

        sampled_pairs = self.sample_items(
            task_question_pairs, n, shuffle=shuffle, with_replacement=with_replacement
        )
        sampled_questions = {task.task_name: [] for task in self.dataset.obj.tasks}
        for task, q_id in sampled_pairs:
            sampled_questions[task.task_name].append(q_id)

        return sampled_questions

    def get_question_ids(self, bootstrap: bool = False, n: int = None) -> Dict[str, List[str]] | List[str]:
        """
        Get question IDs from dataset. If multitask, return a dict of task_name to question IDs.
        Parameters:
            bootstrap (bool): Whether to sample with replacement.
            n (int): Number of question IDs to return if bootstrapping. By default, returns 80% of available question IDs.
        """
        from rank_no_eval.io import load_question_ids

        if self.dataset.obj.is_multitask:
            if bootstrap:
                task_wise_question_ids = self.sample_multitask_questions(
                    n=n, shuffle=True, with_replacement=True
                )
            else:
                task_wise_question_ids = {
                    task.task_name: load_question_ids(self.dataset, task=task)
                    for task in self.dataset.obj.tasks
                }
            return task_wise_question_ids
        else:
            question_ids = load_question_ids(self.dataset)
            if bootstrap:
                question_ids = self.sample_items(
                    question_ids, n=n, shuffle=True, with_replacement=True
                )
            return question_ids
