from .utils import QueryMode, ClientType, Datasets, Task
from .io import *
from .client import VLLMClient, LiteLLMClient
from .utils import logger
from .query import BBHQueries, get_dataset_query_method


class ModelPrompter:
    def __init__(
        self,
        model_id: str,
        dataset: str,
        query_mode: QueryMode,
        client_type: ClientType,
        task: Task = None,
    ):
        self.model_id = model_id
        self.dataset = Datasets(dataset)
        self.query_mode = query_mode
        self.client_type = client_type
        if task:
            assert task in self.dataset.obj.tasks, "Task not in dataset tasks."
        self.task = task
        self.client = None

    def _load_samples_model_answers(self, rerun: bool = False):
        samples = load_freeform_dataset(dataset=self.dataset, task=self.task)
        model_answers = load_answers(
            self.model_id, query_mode=self.query_mode, dataset=self.dataset, task=self.task
        )

        if rerun or model_answers is None:
            return samples, model_answers

        # limit samples to questions that have no answer (None or "")

        # questions that received an answer
        questions_with_answer = [a["question_id"] for a in model_answers]
        # questions that received an empty answer
        questions_with_empty_answer = [
            a["question_id"] for a in model_answers if a["answer"] == ""
        ]
        logger.debug(f"{len(questions_with_empty_answer)} empty answers.")

        # filter out questions that have an answer
        samples = samples.filter(
            lambda x: x["question_id"] not in questions_with_answer
            or x["question_id"] in questions_with_empty_answer
        )
        # remove empty answers
        model_answers = [
            a
            for a in model_answers
            if a["question_id"] not in questions_with_empty_answer
        ]

        return samples, model_answers

    def _questions_missing_answers(
        self, samples: Dataset, model_answers: List[dict]
    ) -> bool:
        sample_ids = set(samples["question_id"])
        answer_ids = set([a["question_id"] for a in model_answers])
        diff = sample_ids - answer_ids
        return len(diff) > 0

    def _missing_answers(self, samples: Dataset, model_answers: List[dict]) -> bool:
        # if there are no answers at all
        if model_answers is None:
            missing_answers = True
        # if some answers are missing
        elif self._questions_missing_answers(samples, model_answers):
            missing_answers = True
        else:
            missing_answers = False
        return missing_answers

    def set_task(self, task):
        self.task = task
        if hasattr(self, 'query_method') and isinstance(self.query_method, BBHQueries):
            self.query_method.task = task

    def run(
        self, rerun: bool = False, run_async: bool = False, sycophant: bool = False
    ):
        samples, model_answers = self._load_samples_model_answers(rerun=rerun)

        if self._missing_answers(samples, model_answers) or rerun:
            logger.info(f"Collecting model answers for {len(samples)} samples...")
            query_method = get_dataset_query_method(
                dataset=self.dataset,
                query_mode=self.query_mode,
                sycophant=sycophant,
                task=self.task,
            )
            if self.client_type == ClientType.VLLM:
                if self.client is None:
                    self.client = VLLMClient(model_id=self.model_id, query_method=query_method)
                else:
                    self.client.update_query_method(query_method)
                answers = self.client.collect_model_answers(samples)
            elif self.client_type == ClientType.LITELLM:
                client = LiteLLMClient(
                    model_id=self.model_id, query_method=query_method
                )
                answers = client.collect_model_answers(samples, run_async=run_async)
            ids_answers = [
                {"question_id": sample["question_id"], "answer": answer}
                for sample, answer in zip(samples, answers)
            ]
            if model_answers is None or rerun:
                model_answers = ids_answers
            else:
                model_answers.extend(ids_answers)
            save_answers(
                self.model_id,
                model_answers,
                query_mode=self.query_mode,
                dataset=self.dataset,
                task=self.task,
            )
