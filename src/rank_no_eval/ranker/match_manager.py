import pandas as pd
from typing import List, Union
from pathlib import Path

from .base import MatchManager
from rank_no_eval.utils import MatchType, Datasets, Task, logger, model_set_id


class ExactMatchManager(MatchManager):
    def __init__(self):
        super().__init__(match_type=MatchType.EXACT)

    def match_file(self, dataset: Datasets, model_id: str, task: Task = None) -> Path:
        if task:
            return (
                self.matches_base_dir
                / dataset.value
                / task.task_name
                / f"{model_id.replace('/', '--')}.csv"
            )
        else:
            return (
                self.matches_base_dir
                / dataset.value
                / f"{model_id.replace('/', '--')}.csv"
            )

    def load_matches(
        self, dataset: Datasets, model_id: str, task: Task = None
    ) -> Union[List, None]:
        mf = self.match_file(dataset, model_id, task=task)
        if mf.exists():
            logger.info(f"Loading matches from file for {model_id}...")
            # load emtpy strings, not nan
            return (
                pd.read_csv(
                    mf,
                    dtype={"question_id": str, "extracted_answer": str, "answer": str},
                    keep_default_na=False,
                )
                .replace("EMPTY_STRING", "")  # map "EMPTY_STRING" back to empty strings
                .replace("NULL", None)  # map 'NULL' back to nan
            )
        else:
            return None

    def save_matches(
        self, matches: pd.DataFrame, dataset: Datasets, model_id: str, task: Task = None
    ):
        mf = self.match_file(dataset, model_id, task=task)
        # create directories if they do not exist
        mf.parent.mkdir(parents=True, exist_ok=True)
        # map empty strings to "EMPTY_STRING" and nan to 'NULL' when saving
        matches.replace("", "EMPTY_STRING").to_csv(mf, index=False, na_rep='NULL')
        logger.info(f"Model matches saved to {mf}.")


class AnswerMatchManager(MatchManager):
    def __init__(self, judge_id: str, with_ground_truth: bool = True):
        self.judge_id = judge_id
        super().__init__(
            match_type=MatchType.AM if with_ground_truth else MatchType.AM_NO_GT
        )

    def match_file(self, dataset: Datasets, model_id: str, task: Task = None) -> Path:
        if task:
            return (
                self.matches_base_dir
                / self.judge_id.replace("/", "--")
                / dataset.value
                / task.task_name
                / f"{model_id.replace('/', '--')}.csv"
            )
        else:
            return (
                self.matches_base_dir
                / self.judge_id.replace("/", "--")
                / dataset.value
                / f"{model_id.replace('/', '--')}.csv"
            )

    def load_matches(
        self, dataset: Datasets, model_id: str, task: Task = None
    ) -> Union[List, None]:
        mf = self.match_file(dataset, model_id, task=task)
        if mf.exists():
            logger.info(f"Loading matches from file for {model_id}...")
            df = pd.read_csv(mf, dtype={"question_id": str})
            return df
        else:
            return None

    def save_matches(
        self,
        matches: pd.DataFrame,
        dataset: Datasets,
        model_id: str,
        task: Task = None,
    ):
        mf = self.match_file(dataset, model_id, task=task)
        # create directories if they do not exist
        mf.parent.mkdir(parents=True, exist_ok=True)
        matches.to_csv(mf, index=False)
        logger.info(f"Model matches saved to {mf}.")


class PairwiseMatchManager(MatchManager):
    def __init__(self):
        super().__init__(match_type=MatchType.PAIRWISE)

    def match_file(
        self,
        dataset: Datasets,
        model_id: str,
        models: List[str],
        what: str,
        answers_only: bool,
    ) -> Path:
        # I don't like passing the models and what here - maybe use a ranking config to store info like this?
        models_id = model_set_id(models, n=8)
        if answers_only:
            what += "_answers_only"
        return (
            self.matches_base_dir
            / models_id
            / dataset.value
            / f"{what}--{model_id.replace('/', '--')}.csv"
        )

    def load_matches(
        self,
        dataset: Datasets,
        model_id: str,
        models: List[str],
        what: str,
        answers_only: bool,
    ) -> Union[List, None]:
        mf = self.match_file(dataset, model_id, models, what, answers_only)
        if mf.exists():
            logger.info(f"Loading matches...")
            df = pd.read_csv(mf, dtype={"question_id": str})
            return df.to_dict(orient="records")
        else:
            return None

    def save_matches(
        self,
        matches: pd.DataFrame,
        dataset: Datasets,
        model_id: str,
        models: List[str],
        what: str,
        answers_only: bool,
    ):
        mf = self.match_file(dataset, model_id, models, what, answers_only)
        # create directories if they do not exist
        mf.parent.mkdir(parents=True, exist_ok=True)
        if mf.exists():
            # append to existing file
            matches.to_csv(mf, index=False, mode="a", header=False)
        else:
            # create new file
            matches.to_csv(mf, index=False)
        logger.info(f"Matches saved to {mf}.")
