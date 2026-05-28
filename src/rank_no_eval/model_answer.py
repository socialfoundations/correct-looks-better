from typing import Dict
import pandas as pd
from datasets import Dataset

from .io import load_freeform_dataset, load_answers, load_question_ids
from .utils import QueryMode, Datasets, logger


class ModelAnswer:
    def __init__(
        self,
        model_id: str,
        question_id: str,
        question: str,
        answer: str,
        metadata: Dict = None,
    ):
        self.model_id = model_id
        self.question_id = question_id
        self.question = question
        self.answer = answer
        self.metadata = metadata


class ModelAnswerStore:
    def __init__(
        self,
        model_id: str,
        preloaded_dataset: Dataset,
        dataset: Datasets,
        query_mode: QueryMode = QueryMode.FREEFORM,
        with_metadata: bool = False,
    ):
        self.model_id = model_id
        self.preloaded_dataset = preloaded_dataset
        self.answer_store: Dict[str, ModelAnswer] = self._load_answers(
            dataset, query_mode, with_metadata
        )

    def _load_answers(
        self, dataset: Datasets, query_mode: QueryMode, with_metadata: bool
    ) -> Dict[str, ModelAnswer]:
        logger.debug(f"Initializing ModelAnswerStore for model {self.model_id}...")
        questions = self.preloaded_dataset.to_pandas().set_index("question_id")
        if dataset.obj.is_multitask:
            answers = []
            for task in dataset.obj.tasks:
                answers.extend(load_answers(
                    model_id=self.model_id,
                    dataset=dataset,
                    query_mode=query_mode,
                    with_metadata=with_metadata,
                    task=task,
                ))
            answers = pd.DataFrame(answers)
        else:
            answers = pd.DataFrame(
                load_answers(
                    model_id=self.model_id,
                    dataset=dataset,
                    query_mode=query_mode,
                    with_metadata=with_metadata,
                ),
            )
        # filter answers
        answers = answers[
            answers["question_id"].astype(str).isin(load_question_ids(dataset))
        ]
        answer_dict = {}
        for i, a in answers.iterrows():
            question = questions.loc[str(a["question_id"]), "question"]
            question_id = str(a["question_id"])
            answer_dict[question_id] = ModelAnswer(
                model_id=self.model_id,
                question_id=question_id,
                question=question,
                answer=a["answer"],
                metadata=a["metadata"] if with_metadata else None,
            )
        logger.debug(f"Loaded {i + 1} answers for model {self.model_id}.")
        return answer_dict

    def get(self, question_id: str) -> ModelAnswer:
        if question_id not in self.answer_store:
            raise KeyError(f"Question ID {question_id} not found in the answer store.")
        return self.answer_store[question_id]

    def get_random(self) -> ModelAnswer:
        import random

        question_id = random.choice(list(self.answer_store.keys()))
        return self.answer_store[question_id]
