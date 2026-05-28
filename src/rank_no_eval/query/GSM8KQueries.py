from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.utils import QueryMode, Datasets
from rank_no_eval.config import PROMPTS_DIR

class GSM8KQueries(BaseQueryMethod):
    def __init__(self, query_mode: QueryMode, sycophant: bool = False):
        self.sycophant = sycophant
        self.query_mode = query_mode
        self.env = Environment(
            loader=FileSystemLoader(
                PROMPTS_DIR / query_mode.value / Datasets.GSM8K.value
            )
        )
        self.max_tokens = 1024
        self.temperature = 0.0
        self.max_model_len = 4096 # prompt + response max length
        self.oss_reasoning_effort = "Low"
        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("question.jinja2")
        question = sample["question"]
        prompt = question_template.render(question=question)
        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        # no system prompt was used in GSM8K
        system_prompt = ""
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