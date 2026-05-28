import json
from enum import Enum
from typing import List

from rank_no_eval.client import VLLMClient
from rank_no_eval.query import EchoDetectionQueries
from rank_no_eval.utils import logger


class EchoDetectorJudgeResponse(Enum):
    ECHO = "ECHO"
    NO_ECHO = "NO_ECHO"
    EMPTY = "EMPTY"
    PARSE_ERROR = "PARSE_ERROR"


class EchoDetector:
    """Detects echo failure modes in model QA pairs using a VLLM judge model."""

    def __init__(self, judge_model: str = "openai/gpt-oss-120b"):
        self.judge_model = judge_model
        self.query_method = EchoDetectionQueries()
        self.client = VLLMClient(model_id=judge_model, query_method=self.query_method)

    def detect(self, qa_pairs: List[str]) -> List[EchoDetectorJudgeResponse]:
        """
        Classify each QA pair string as 'ECHO' or 'NO_ECHO'.

        Args:
            qa_pairs: List of raw QA pair strings to evaluate.

        Returns:
            List of EchoDetectorJudgeResponse labels.
        """
        samples = [{"qa_pair": qa_pair} for qa_pair in qa_pairs]
        raw_answers = self.client.collect_model_answers(samples)

        results = []
        for i, answer in enumerate(raw_answers):
            if not answer:
                logger.error(f"Empty response for sample {i}.")
                results.append(EchoDetectorJudgeResponse.EMPTY)
                continue
            try:
                parsed = json.loads(answer)
                results.append(EchoDetectorJudgeResponse(parsed["echo"]))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse response for sample {i}: {e!r}. Raw: {answer!r}")
                results.append(EchoDetectorJudgeResponse.PARSE_ERROR)
        return results
