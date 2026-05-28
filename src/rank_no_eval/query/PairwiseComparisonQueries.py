from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.config import PROMPTS_DIR
from rank_no_eval.utils import logger


class PairwiseComparisonQueries(BaseQueryMethod):
    def __init__(self, variant: str = "qa_qa", answers_only: bool = False):
        # variant: comparing question-answer pairs (qa_qa) or answer pairs (a_a)
        assert variant in ["qa_qa", "a_a"]
        if answers_only:
            variant = "answers_only"
            logger.warning("Overriding variant to 'answers_only'.")
        self.answers_only = answers_only
        self.env = Environment(
            loader=FileSystemLoader(PROMPTS_DIR / "pairwise_comparison" / variant)
        )
        self.oss_reasoning_effort = "High"
        self.max_model_len = 16384 # prompt + response max length
        self.max_tokens = 128 # response is just {"winner": 1}
        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("prompt.jinja2")
        prompt = question_template.render(qap=[sample["p1"], sample["p2"]])
        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        return self.env.get_template("system_prompt.jinja2").render()

    @property
    def answer_format(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ranking_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {"type": "integer", "enum": [1, 2]},
                    },
                    "required": ["winner"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @property
    def completion_kwargs(self):
        return {
            "response_format": self.answer_format,
        }
