from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.config import PROMPTS_DIR
from rank_no_eval.utils import Datasets, QueryMode, Task



class BBHQueries(BaseQueryMethod):
    def __init__(self, query_mode: QueryMode, task: Task, sycophant: bool = False):
        self.sycophant = sycophant
        self.query_mode = query_mode
        self.task = task
        self.env = Environment(
            loader=FileSystemLoader(
                PROMPTS_DIR / self.query_mode.value / Datasets.BBH.value
            )
        )
        self.temperature = 0.0
        self.max_tokens = 2048
        self.max_model_len = 4096  # prompt + response max length
        self.oss_reasoning_effort = "Low"
        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("question.jinja2")
        question = sample["question"]
        if self.query_mode == QueryMode.MCQ:
            choices = sample.get("choices", None)   
            prompt = question_template.render(question=question, choices=choices)
        elif self.query_mode == QueryMode.FREEFORM:
            prompt = question_template.render(question=question)
        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        cot_prompt = self.env.get_template(f"cot-prompts/{self.task.task_name}.txt").render()
        if self.sycophant:
            local_env = Environment(
                loader=FileSystemLoader(PROMPTS_DIR / self.query_mode.value)
            )
            sycophancy_instruction = local_env.get_template(
                "sycophancy_instruction.jinja2"
            ).render()
            cot_prompt += "\n" + sycophancy_instruction
        return cot_prompt

    @property
    def completion_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }