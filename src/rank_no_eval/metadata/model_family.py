from enum import Enum


class ModelFamily(Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    OTHER = "other"


_model_provider_mapping = {
    "openai": ModelFamily.GPT,
    "azure": ModelFamily.GPT,
    "anthropic": ModelFamily.CLAUDE,
    "meta-llama": ModelFamily.LLAMA,
    "qwen": ModelFamily.QWEN,
    "microsoft": ModelFamily.PHI,
    "deepseek-ai": ModelFamily.DEEPSEEK,
}
_model_provider_mapping = {k.lower(): v for k, v in _model_provider_mapping.items()}

# model_id starts with model provider
_rules = [
    (lambda model_id, f_key=f_key: model_id.startswith(f_key), f_val)
    for f_key, f_val in _model_provider_mapping.items()
]
# model_id contains model family name
_rules += [(lambda model_id, val=mf.value: val in model_id, mf) for mf in ModelFamily]


def get_family(model_id: str) -> ModelFamily:
    model_id = model_id.lower()
    for rule, family in _rules:
        if rule(model_id):
            return family
    return ModelFamily.OTHER
