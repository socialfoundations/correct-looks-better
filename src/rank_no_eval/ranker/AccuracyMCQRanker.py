from typing import List, Tuple
import pandas as pd


from rank_no_eval.io import load_answers, load_freeform_dataset
from rank_no_eval.eval_object import ModelAccuracy
from rank_no_eval.utils import logger, QueryMode, Task, TaskType
from .base import BaseRanker
from .mcq_utils import extract_chosen_answer_letter
from .match_manager import ExactMatchManager
from .match_aggregator import AccuracyMatchAggregator


class AccuracyMCQRanker(BaseRanker):
    def __init__(self, dataset: str, eval_objects: List[ModelAccuracy] = None):
        self.query_mode = QueryMode.MCQ
        self.aggregator = AccuracyMatchAggregator(eval_objects)

        super().__init__(
            eval_objects=eval_objects,
            match_manager=ExactMatchManager(),
            dataset=dataset,
        )

    def create_matches_from_answers(self, answers: List[str], task: Task = None):
        data = [
            {
                **d,
                "extracted_answer": "" if d["answer"] == "" else extract_chosen_answer_letter(
                    d["answer"], self.dataset, task=task
                ),
            }
            for d in answers
        ]

        matches_df = pd.DataFrame(data, columns=["question_id", "extracted_answer"])
        # question id as string
        matches_df["question_id"] = matches_df["question_id"].astype(str)
        # add true answers
        samples = load_freeform_dataset(dataset=self.dataset, task=task)
        true_answers_df = samples.to_pandas()[["question_id", "answer"]]
        # question id as string
        true_answers_df["question_id"] = true_answers_df["question_id"].astype(str)
        matches_df = pd.merge(
            matches_df, true_answers_df, on="question_id", how="inner"
        )
        return matches_df

    def create_matches(self, model_id: str, task: Task = None):
        # load answers
        if task is not None:
            query_mode = (
                QueryMode.FREEFORM
                if task.task_type != TaskType.MULTIPLE_CHOICE
                else QueryMode.MCQ
            )
        else:
            query_mode = self.query_mode  # default to MCQ
        answers = load_answers(model_id, query_mode, self.dataset, task=task)
        if answers is None:
            logger.error(
                f"Answers file not found for {model_id}. Skipping ranking."
            )  # TODO: raise exception?
            return
        # create matches
        matches = self.create_matches_from_answers(answers, task=task)
        # save matches
        self.match_manager.save_matches(
            matches=matches,
            dataset=self.dataset,
            model_id=model_id,
            task=task,
        )

        return matches

    def get_matches(
        self, model_id: str, task: Task = None, question_ids: List[str] = []
    ) -> List[Tuple[str, str]]:
        matches = self.match_manager.load_matches(
            dataset=self.dataset, model_id=model_id, task=task
        )
        if matches is None:
            logger.info(
                f"Matches file not found for {model_id}. Constructing matches from answers..."
            )
            matches = self.create_matches(model_id, task=task)
        # filter matches
        matches = matches[matches["question_id"].isin(question_ids)]
        # extract matches
        matches = zip(matches["extracted_answer"], matches["answer"])
        return matches

    def rank(self, bootstrap: bool = False, n: int = None):
        logger.info("Ranking models...")
        question_ids = self.get_question_ids(bootstrap=bootstrap, n=n)
        if self.dataset.obj.is_multitask:
            logger.info("Detected multitask dataset.")
            for task in self.dataset.obj.tasks:
                logger.info(f"Ranking for task: {task.task_name}")
                matches = {
                    ma.id: self.get_matches(ma.id, task=task, question_ids=question_ids.get(task.task_name, []))
                    for ma in self.eval_objects
                }
                self.aggregator.aggregate(matches, task=task.task_name)
        else:
            logger.info("Ranking for single-task dataset.")
            matches = {
                ma.id: self.get_matches(ma.id, question_ids=question_ids)
                for ma in self.eval_objects
            }
            self.aggregator.aggregate(matches)
        # sort eval objects by score
        self.eval_objects = sorted(
            self.eval_objects, key=lambda x: x.score, reverse=True
        )

    def model_ranking(self) -> Tuple[str, float]:
        """Return model IDs and their score in descending order."""
        ranking = [(ma.model_id, ma.score.score) for ma in self.eval_objects]
        return ranking
