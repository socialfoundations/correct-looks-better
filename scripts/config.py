from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "out"
BOOTSTRAP_DIR = OUT_DIR / "bootstrap"

# create the bootstrap directory if it doesn't exist
BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)


model_name_mapping = {
    "alpindale/WizardLM-2-8x22B": "WizardLM-2 8x22B",
    "openai/gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "openai/gpt-4o-2024-08-06": "GPT-4o",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/o3": "o3",
    "openai/o4-mini": "o4-mini",
    "anthropic/claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
    "anthropic/claude-opus-4-20250514": "Claude Opus 4",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "DeepSeek R1 Distill Llama 70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek R1 Distill Qwen 32B",
    "google/gemma-3-1b-it": "Gemma 3 1B",
    "microsoft/phi-4": "Phi-4",
    "microsoft/Phi-4-mini-instruct": "Phi-4 Mini",
    "meta-llama/Llama-3.1-8B": "Llama 3.1 8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama 3.1 70B Instruct",
    "Qwen/Qwen3-0.6B": "Qwen3 0.6B",
    "Qwen/Qwen3-32B": "Qwen3 32B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3 30B A3B Instruct",
    "Qwen/Qwen2.5-72B": "Qwen2.5 72B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5 72B Instruct",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen3 4B Thinking",
}
