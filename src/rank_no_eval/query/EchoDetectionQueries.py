from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.config import PROMPTS_DIR


class EchoDetectionQueries(BaseQueryMethod):
    def __init__(self):
        self.max_tokens = 1024
        self.oss_reasoning_effort = "Medium"
        self.env = Environment(loader=FileSystemLoader(PROMPTS_DIR / "echo_detection"))
        self.max_model_len = 8192 # prompt + response max length

        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("prompt.jinja2")
        prompt = question_template.render(qa_pair=sample["qa_pair"])
        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        return self.env.get_template("system_prompt.jinja2").render()
    
    @property
    def answer_format(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "echo_detection_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "echo": {"type": "string", "enum": ["ECHO", "NO_ECHO"]},
                    },
                    "required": ["echo"],
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
