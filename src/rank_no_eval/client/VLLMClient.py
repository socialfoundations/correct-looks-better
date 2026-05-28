from vllm import LLM, SamplingParams, TokensPrompt
from vllm.sampling_params import StructuredOutputsParams
import torch
from typing import List
from datasets import Dataset
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    HarmonyError
)

from rank_no_eval.utils import REPRODUCIBLE_RANKING, SEED
from rank_no_eval.utils import logger
from rank_no_eval.query.base import BaseQueryMethod
from .base import BaseClient


OSS_MODELS = {
    "openai/gpt-oss-120b", "openai/gpt-oss-20b"
}


class VLLMClient(BaseClient):

    def __init__(self, model_id: str, query_method: BaseQueryMethod):
        super().__init__(model_id, query_method)
        self.model, self.sampler = self.load_model_sampler()

    def load_model_sampler(self):
        max_model_len = getattr(self.query_method, "max_model_len", None)
        max_tokens = getattr(self.query_method, "max_tokens", None)
        temperature = getattr(self.query_method, "temperature", 1)

        # Output restricting parameters
        json_schema = getattr(self.query_method, "answer_format", None)
        if json_schema is not None:
            json_schema = json_schema.get("json_schema").get("schema")
            logger.debug(f"Json schema: {json_schema}")
        # mutually exclusive with json_schema - if both are present, json_schema takes precedence
        allowed_regex = getattr(self.query_method, "allowed_regex", None)

        model = LLM(
            model=self.model_id,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.bfloat16,
            max_model_len=max_model_len,
            tokenizer_mode="mistral" if "mistral" in self.model_id else "auto",
            enforce_eager=True,
        )
        has_think = "<think>" in model.get_tokenizer().get_vocab()
        is_oss = self.model_id in OSS_MODELS

        sampler = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens if not is_oss else None,
            seed=SEED if REPRODUCIBLE_RANKING == 1 else None,
            stop_token_ids=(
                self._oss_harmony_stop_token_ids()
                if is_oss else None
            ),
            structured_outputs=(
                StructuredOutputsParams(json=json_schema) if json_schema is not None
                else StructuredOutputsParams(regex=allowed_regex) if allowed_regex is not None
                else None
            ),
        )
        if max_tokens and is_oss:
            logger.warning(f"Suppressed max_tokens={max_tokens} for OSS model {self.model_id} to enable reasoning traces.")
        if max_tokens and has_think:
            logger.warning(f"Model {self.model_id} appears to support reasoning (has '<think>' token), but max_tokens is set to {max_tokens}. This may truncate reasoning traces.")
        if json_schema is None and allowed_regex is not None and has_think:
            logger.warning(
                f"Model {self.model_id} appears to support reasoning (has '<think>' token), "
                f"but output is constrained to regex '{allowed_regex}'. Reasoning traces will be suppressed."
            )
        logger.info(f"Model {self.model_id} loaded successfully.")
        return model, sampler
    
    def _oss_harmony_stop_token_ids(self):
        # Harmony stop tokens (pass to sampler so they won't be included in output)
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        return stop_token_ids
    
    def _oss_structured_input_instruction(self):
        # Based on: https://cookbook.openai.com/articles/openai-harmony#structured-output
        import json

        format_name = self.query_method.answer_format["json_schema"]["name"]
        schema = self.query_method.answer_format["json_schema"]["schema"]
        instruction_text = f"""
        # Response Formats
        ## {format_name}
        {json.dumps(schema)}
        """
        return instruction_text

    def _oss_harmony_encoding(self, encoding, system_message: str, user_message: str):
        dev_message = system_message
        if hasattr(self.query_method, "answer_format"):
            dev_message += self._oss_structured_input_instruction()

        reasoning_effort = getattr(self.query_method, "oss_reasoning_effort", "Medium")

        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(reasoning_effort)),
                Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions(dev_message)),
                Message.from_role_and_content(Role.USER, user_message),
            ]
        )

        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        return TokensPrompt(prompt_token_ids=prefill_ids)
    

    def prepare_prompts(self, samples: Dataset) -> List[str]:
        logger.debug("Preparing prompts...")
        is_oss = self.model_id in OSS_MODELS

        if is_oss:
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            def build_prompt(sample):
                cot = self.query_method.get_system_prompt(sample)
                prompt = self.query_method.get_prompt(sample)
                return self._oss_harmony_encoding(encoding, system_message=cot, user_message=prompt)
            prompts = [build_prompt(sample) for sample in samples]
            logger.debug(f"First prompt:\n\n{encoding.decode(prompts[0]['prompt_token_ids'])}")
        else:
            def build_prompt(sample):
                cot = self.query_method.get_system_prompt(sample)
                prompt = self.query_method.get_prompt(sample)
                return cot + "\n\n" + prompt
            prompts = [build_prompt(sample) for sample in samples]
            logger.debug(f"First prompt:\n\n{prompts[0]}")
        return prompts
    
    def outputs_to_answers(self, outputs: List[str]) -> List[str]:
        if self.model_id in OSS_MODELS:
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            output_tokens = [output.outputs[0].token_ids for output in outputs]

            def decode_output(token_ids):
                try:
                    parsed_entry = encoding.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
                    return parsed_entry
                except HarmonyError as e:
                    logger.error(f"Failed to decode output: {e}")
                    return []

            entries = [decode_output(token_ids) for token_ids in output_tokens]
            logger.debug(f"First entry: {entries[0]}")
            final_texts = []
            for entry in entries:
                final_text, analysis_text = "", ""
                if len(entry) == 0:
                    logger.error("Empty entry.")
                elif len(entry) > 2:
                    logger.debug(f"Entry has {len(entry)} messages. This is more than expected (2: final and analysis).")
                    logger.debug(f"Entry channels: {[message.channel for message in entry]}")
                for message in entry:
                    content = message.content[0]
                    if message.channel == "final" and content.text is not None:
                        final_text = content.text
                final_texts.append(final_text)

            return final_texts

        return [output.outputs[0].text for output in outputs]

    @torch.no_grad()
    def collect_model_answers(self, samples: Dataset) -> List[str]:
        prompts = self.prepare_prompts(samples)
        # Extract answers from the model
        outputs = self.model.generate(prompts, sampling_params=self.sampler)
        model_answers = self.outputs_to_answers(outputs)
        return model_answers
    
    def update_query_method(self, query_method):
        self.query_method = query_method
