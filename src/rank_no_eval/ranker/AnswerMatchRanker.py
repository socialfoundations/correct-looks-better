from typing import List, Tuple

import pandas as pd

from .base import BaseRanker
from rank_no_eval.eval_object import ModelAnswerMatch
from .match_manager import AnswerMatchManager
from rank_no_eval.utils import QueryMode, logger, ClientType, Datasets, Task
from rank_no_eval.io import load_answers, load_freeform_dataset
from rank_no_eval.client import VLLMClient, LiteLLMClient
from rank_no_eval.query import AnswerMatchingQueries
from .match_aggregator import AnswerMatchAggregator


class AnswerMatchRanker(BaseRanker):

    def __init__(
        self,
        dataset: str,
        judge_model: str,
        client_type: ClientType,
        eval_objects: List[ModelAnswerMatch] = None,
        with_ground_truth: bool = True,
    ):
        self.judge_model = judge_model
        self.client_type = client_type
        self.with_ground_truth = with_ground_truth
        dset = Datasets(dataset)
        self.query_method = AnswerMatchingQueries(dataset_name=dset.obj.dataset_name, 
                                                  with_ground_truth=with_ground_truth)
        self.aggregator = AnswerMatchAggregator(eval_objects)
        self.client = None  # Will be initialized on first use

        super().__init__(
            eval_objects=eval_objects,
            match_manager=AnswerMatchManager(judge_id=judge_model, with_ground_truth=with_ground_truth),
            dataset=dataset,
        )

    def load_model_true_answers(self, model_id: str, task: Task = None) -> pd.DataFrame:
        # load answers
        answers = load_answers(model_id, QueryMode.FREEFORM, self.dataset, task=task)
        if answers is None:
            logger.error(f"Answers file not found for {model_id}. Skipping ranking.")
            return
        answers_df = pd.DataFrame(answers, columns=["question_id", "answer"], dtype=str)
        # rename answer column to predicted answer
        answers_df = answers_df.rename(columns={"answer": "predicted_answer"})
        # load true answers
        true_answers = load_freeform_dataset(dataset=self.dataset, task=task)
        true_answers_df = true_answers.to_pandas()[
            ["question_id", "answer", "question"]
        ]
        answers_df = pd.merge(
            answers_df, true_answers_df, on="question_id", how="inner"
        )
        return answers_df

    def _get_or_create_client(self):
        """Get or create the client instance."""
        if self.client is None:
            if self.client_type == ClientType.VLLM:
                self.client = VLLMClient(
                    model_id=self.judge_model, query_method=self.query_method
                )
            elif self.client_type == ClientType.LITELLM:
                self.client = LiteLLMClient(
                    model_id=self.judge_model, query_method=self.query_method
                )
        return self.client

    def judge_answers(self, answers_df: pd.DataFrame, run_async: bool = False):
        logger.info("Running Answer Matching Judge...")
        # load client
        client = self._get_or_create_client()
        # collect model answers
        if self.client_type == ClientType.VLLM:
            judgments = client.collect_model_answers(answers_df.to_dict("records"))
        elif self.client_type == ClientType.LITELLM:
            judgments = client.collect_model_answers(
                answers_df.to_dict("records"), run_async=run_async
            )
        return judgments

    def create_matches_from_answers_and_judgments(
        self, answers_df: pd.DataFrame, judgments: List[Tuple]
    ):
        matches = []
        for i, (predicted_answer, answer, judge_answer) in enumerate(
            zip(answers_df["predicted_answer"], answers_df["answer"], judgments)
        ):
            matches.append(
                {
                    "question_id": answers_df["question_id"].iloc[i],
                    "judge_answer": judge_answer,
                    "predicted_answer": predicted_answer,
                    "answer": answer,
                }
            )
        return pd.DataFrame(matches)

    def create_matches(self, model_id: str, run_async: bool = False, task: Task = None):
        answers_df = self.load_model_true_answers(model_id, task=task)
        # judge answers
        judgments = self.judge_answers(answers_df, run_async=run_async)
        # create matches
        matches = self.create_matches_from_answers_and_judgments(answers_df, judgments)
        # save matches
        self.match_manager.save_matches(
            matches=matches,
            dataset=self.dataset,
            model_id=model_id,
            task=task,
        )
        # extract matches
        return matches

    def get_matches(
        self,
        model_id: str,
        run_async: bool = False,
        task: Task = None,
        question_ids: List[str] = [],
    ) -> pd.DataFrame:
        matches = self.match_manager.load_matches(
            dataset=self.dataset, model_id=model_id, task=task
        )
        if matches is None:
            logger.info(
                f"Matches file not found for {model_id}. Constructing matches from answers..."
            )
            matches = self.create_matches(model_id, run_async=run_async, task=task)
        # filter matches
        matches = matches[matches["question_id"].isin(question_ids)]
        return matches

    def rank(self, run_async: bool = False, bootstrap: bool = False, n: int = None):
        logger.info("Ranking models...")
        question_ids = self.get_question_ids(bootstrap=bootstrap, n=n)
        if self.dataset.obj.is_multitask:
            logger.info("Detected multitask dataset.")
            for task in self.dataset.obj.tasks:
                logger.info(f"Ranking for task: {task.task_name}")
                matches = {
                    ma.id: self.get_matches(
                        ma.id,
                        run_async=run_async,
                        task=task,
                        question_ids=question_ids.get(task.task_name, []),
                    )
                    for ma in self.eval_objects
                }
                self.aggregator.aggregate(matches, task=task.task_name)
        else: 
            matches = {
                ma.id: self.get_matches(
                    ma.id, run_async=run_async, question_ids=question_ids
                )
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
