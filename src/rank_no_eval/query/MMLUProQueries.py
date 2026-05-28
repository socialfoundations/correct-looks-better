from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.utils import QueryMode, Datasets
from rank_no_eval.config import PROMPTS_DIR


class MMLUProQueries(BaseQueryMethod):
    # Based on: https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main
    def __init__(self, query_mode: QueryMode, sycophant: bool = False):
        self.sycophant = sycophant
        self.query_mode = query_mode
        self.env = Environment(
            loader=FileSystemLoader(
                PROMPTS_DIR / query_mode.value / Datasets.MMLU_PRO.value
            )
        )
        self.max_tokens = 2048
        self.temperature = 0.0
        self.max_model_len = 4096 # prompt + response max length
        self.oss_reasoning_effort = "Low"
        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("question.jinja2")
        question = sample["question"]
        if self.query_mode == QueryMode.MCQ:
            options = [
                {"letter": chr(65 + i), "text": option}
                for i, option in enumerate(sample["options"])
            ]
            prompt = question_template.render(question=question, options=options)
        elif self.query_mode == QueryMode.FREEFORM:
            prompt = question_template.render(question=question)

        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        subject = sample["category"]
        system_prompt = self.env.get_template("system_prompt.jinja2").render(
            subject=subject
        )
        if self.sycophant:
            local_env = Environment(
                loader=FileSystemLoader(PROMPTS_DIR / self.query_mode.value)
            )
            sycophancy_instruction = local_env.get_template(
                "sycophancy_instruction.jinja2"
            ).render()
            system_prompt += "\n" + sycophancy_instruction

        return system_prompt

    @property
    def completion_kwargs(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
