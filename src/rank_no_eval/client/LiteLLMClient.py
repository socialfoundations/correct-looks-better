from typing import List
from datasets import Dataset
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import backoff
import json

from litellm import acompletion, completion, completion_cost
from litellm import RateLimitError, InternalServerError, ContentPolicyViolationError
from litellm.caching.caching import Cache
import litellm

from rank_no_eval.config import RAW_OUT_DIR

# caching
litellm.cache = Cache(type="disk")

# for faster completion: https://docs.litellm.ai/release_notes/v1.71.1-stable
# litellm.use_aiohttp_transport = True

from rank_no_eval.utils import logger
from rank_no_eval.query.base import BaseQueryMethod
from .base import BaseClient


NO_TEMPERATURE_MODELS = {
    "openai/o3",
    "openai/o4-mini",
}


class LiteLLMClient(BaseClient):
    def __init__(
        self, model_id: str, query_method: BaseQueryMethod, save_raw: bool = False
    ):
        self.total_cost = 0
        self.save_raw = save_raw
        super().__init__(model_id, query_method)

    def _init_raw_answer_store(self):
        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        raw_answer_dir = (
            RAW_OUT_DIR / "litellm" / self.model_id.replace("/", "--") / now
        )
        raw_answer_dir.mkdir(parents=True, exist_ok=True)
        self.raw_answer_file = raw_answer_dir / "responses.jsonl"
        logger.info(f"Raw answers will be saved to {self.raw_answer_file}.")

    def _save_raw_answer(self, response: dict):
        with self.raw_answer_file.open("a") as f:
            f.write(json.dumps(response) + "\n")

    def prepare_message(self, sample: dict) -> dict:
        prompt = self.query_method.get_prompt(sample)
        system_prompt = self.query_method.get_system_prompt(sample)
        usr_content = [
            {"type": "text", "text": prompt},
        ]
        # Anthropic doesn't like empty system prompts
        if system_prompt is not None and system_prompt != "":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_content},
            ]
        else:
            messages = [{"role": "user", "content": usr_content}]
        return messages

    def backoff_handler(details):
        logger.error(
            f"Backing off {details['wait']} seconds after {details['tries']} tries"
        )

    @backoff.on_exception(
        backoff.expo, InternalServerError, max_tries=10, on_backoff=backoff_handler
    )
    @backoff.on_exception(backoff.expo, RateLimitError, on_backoff=backoff_handler)
    async def _query_model(self, sample: dict, async_mode: bool) -> str:
        """Helper method for querying the model, supporting both sync and async modes."""
        message = self.prepare_message(sample)
        kwargs = self.query_method.completion_kwargs
        if self.model_id in NO_TEMPERATURE_MODELS:
            kwargs.pop("temperature", None)  # Remove unsupported param
        try:
            if async_mode:
                response = await acompletion(
                    model=self.model_id,
                    messages=message,
                    caching=True,
                    **kwargs,
                )
            else:
                response = completion(
                    model=self.model_id,
                    messages=message,
                    caching=True,
                    **kwargs,
                )
            self.total_cost += float(completion_cost(response))
        except ContentPolicyViolationError as e:
            logger.error(
                f"Error during {'async ' if async_mode else ''}completion: {e}"
            )
            return ""
        if self.save_raw:
            self._save_raw_answer(response.model_dump(mode="json"))
        content = response["choices"][0]["message"]["content"]
        finish_reason = response["choices"][0]["finish_reason"]
        if finish_reason in [
            "length",
            "max_tokens",
        ]:  # openai and anthropic finish reasons for early truncation
            logger.debug(f"Model ran out of tokens: {response}")
        if content is not None:
            if content == "":
                logger.error(f"Empty string as content. Response: {response}")
            return content
        else:
            logger.error(
                f"No content in {'async ' if async_mode else ''}response: {response}"
            )
            return ""

    def query_model(self, sample: dict) -> str:
        """Synchronous model query."""
        return asyncio.run(self._query_model(sample, async_mode=False))

    async def a_query_model(self, sample: dict) -> str:
        """Asynchronous model query."""
        return await self._query_model(sample, async_mode=True)

    def collect_model_answers(
        self,
        samples: Dataset | List[dict],
        run_async: bool = False,
        save_raw: bool = False,
    ) -> List[str]:
        self.save_raw = save_raw
        if self.save_raw:
            self._init_raw_answer_store()
        # log the first sample for debugging
        logger.debug(
            "First prompt:\n\n"
            + self.query_method.get_system_prompt(samples[0])
            + "\n\n"
            + self.query_method.get_prompt(samples[0])
        )
        if run_async:
            batch_size = 100
            logger.info(
                f"Collecting model answers asynchronously in batches of {batch_size}..."
            )
            model_answers = []
            for i in range(0, len(samples), batch_size):
                logger.info(
                    f"Processing batch {i // batch_size + 1} / {len(samples) // batch_size + 1}"
                )
                if type(samples) == Dataset:
                    stop = min(i + batch_size, len(samples))
                    batch = samples.select(range(i, stop))
                else:
                    batch = samples[i : i + batch_size]
                tasks = [self.a_query_model(sample) for sample in batch]
                batch_answers = asyncio.run(async_tqdm.gather(*tasks))
                model_answers.extend(batch_answers)
                logger.info(f"Cost: ${self.total_cost:.2f}")
        else:
            model_answers = [self.query_model(sample) for sample in tqdm(samples)]
        logger.info(f"Total cost: ${self.total_cost:.2f}")
        return model_answers
