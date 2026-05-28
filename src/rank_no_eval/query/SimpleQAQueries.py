from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.utils import QueryMode
from rank_no_eval.config import PROMPTS_DIR


class SimpleQAQueries(BaseQueryMethod):
    # Based on: https://github.com/openai/simple-evals/blob/main/simple_evals.py
    def __init__(self, query_mode: QueryMode, sycophant: bool = False):
        self.sycophant = sycophant
        assert query_mode == QueryMode.FREEFORM, "SimpleQA only supports freeform mode"
        self.query_mode = query_mode
        self.max_tokens = 2048
        self.max_model_len = 4096 # prompt + response max length
        self.temperature = 1.0

    def get_prompt(self, sample: dict) -> str:
        return sample["question"]

    def get_system_prompt(self, sample: dict) -> str:
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
    def completion_kwargs(self) -> dict:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature}
