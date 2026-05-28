from pathlib import Path

# Paths for data and outputs
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
MMLU_PRO_IDS_PATH = DATA_DIR / "mmlu_pro_human_filtered_ids.txt"
GPQA_DIAMOND_IDS_PATH = DATA_DIR / "gpqa_diamond_human_filtered_ids.txt"
SIMPLE_QA_IDS_PATH = DATA_DIR / "simple_qa_random_ids.txt"
GSM8K_IDS_PATH = DATA_DIR / "gsm8k_ids.txt"
BBH_IDS_PATH = DATA_DIR / "bbh_question_ids.txt"
OUT_DIR = BASE_DIR / "out"
ANSWERS_OUT_DIR = OUT_DIR / "answers"
ANSWERS_WITH_METADATA_OUT_DIR = OUT_DIR / "answers_metadata"
MATCHES_OUT_DIR = OUT_DIR / "matches"
RANKING_OUTPUT_DIR = OUT_DIR / "ranking"
RAW_OUT_DIR = OUT_DIR / "raw"
ECHO_OUT_DIR = OUT_DIR / "echo_detection"

# Ensure output directories exist
ANSWERS_OUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_OUT_DIR.mkdir(parents=True, exist_ok=True)
RANKING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ECHO_OUT_DIR.mkdir(parents=True, exist_ok=True)