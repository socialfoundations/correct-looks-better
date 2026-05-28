from jinja2 import Environment, FileSystemLoader

from .base import BaseQueryMethod
from rank_no_eval.config import PROMPTS_DIR


class AnswerMatchingQueries(BaseQueryMethod):
    def __init__(self, dataset_name: str = "simple_qa", with_ground_truth: bool = True):
        self.with_ground_truth = with_ground_truth
        self.max_tokens = 1024

        if self.with_ground_truth:
            self.oss_reasoning_effort = "High"
            self.env = Environment(loader=FileSystemLoader(PROMPTS_DIR / "answer_matching"))
            self.max_model_len = 8192 if dataset_name == "gsm8k" else 4096 # prompt + response max length
            assert dataset_name in [
                "simple_qa", "gsm8k"
            ], "Supported datasets are: simple_qa, gsm8k" # each dataset has its own grading instructions
            self.dataset_name = dataset_name
        else:
            self.oss_reasoning_effort = "Medium" # without grading instructions, "High" reasoning tends to "overthink" until the model runs out of tokens
            self.env = Environment(loader=FileSystemLoader(PROMPTS_DIR / "answer_matching_no_gt"))
            self.max_model_len = 8192
            self.allowed_regex = "[ABC]" # necessary restriction for non-instruction tuned models that tend to give malformed answers; NOTE: potentially suppresses reasoning traces!!!!
            # all datasets are supported in the no ground truth setting
            self.dataset_name = dataset_name
        super().__init__()

    def get_prompt(self, sample: dict) -> str:
        question_template = self.env.get_template("prompt.jinja2")
        question = sample["question"]
        predicted_answer = sample["predicted_answer"]
        if self.with_ground_truth:
            target = sample["answer"]
            prompt = question_template.render(
                question=question, target=target, predicted_answer=predicted_answer
            )
        else:
            # no target answer in the no ground truth setting
            prompt = question_template.render(
                question=question, predicted_answer=predicted_answer
            )
        return prompt

    def get_system_prompt(self, sample: dict) -> str:
        if self.with_ground_truth:
            # grading instructions for each dataset
            return self.env.get_template(f"{self.dataset_name}.jinja2").render()
        else:
            # no grading instructions
            return ""

    @property
    def completion_kwargs(self):
        return {}
