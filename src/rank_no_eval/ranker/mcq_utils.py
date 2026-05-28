# extracting answer letter from answers to MCQ-style questions
from typing import Optional
import re

from ..utils import Datasets, Task, TaskType


def extract_chosen_answer_letter(answer: str, dataset: Datasets, **kwargs) -> str:
    if dataset == Datasets.MMLU_PRO:
        return _extract_answer_mmlu(answer)
    elif dataset == Datasets.GPQA_DIAMOND:
        return _extract_answer_qpga(answer)
    elif dataset == Datasets.BBH:
        return _extract_answer_bbh(answer, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# MMLU-Pro answer extraction
# based on: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
def _extract_answer_mmlu(text: str) -> Optional[str]:
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return _extract_again(text)


def _extract_again(text: str) -> Optional[str]:
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return _extract_final(text)


def _extract_final(text: str) -> Optional[str]:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


# GPQA-Diamond answer extraction
# based on: https://github.com/idavidrein/gpqa/blob/main/baselines/run_baseline.py#L99
def _extract_answer_qpga(text: str) -> Optional[str]:
    patterns = [
        r"answer is \((.)\)",
        r"Answer: \((.)\)",
        r"answer: \((.)\)",
        r"answer \((.)\)",
        r"\((.)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in ["A", "B", "C", "D"]:
            return match.group(1)
    return None


# BBH answer extraction helper function
# Based on the paper https://arxiv.org/pdf/2210.09261 (page 5, Evaluation Protocol)
# Extraction: string after "the answer is"
# Cases:
#   - Multiple choice (extract a letter)
#   - Binary (extract Yes/No, True/False, etc.)
#   - Open-ended (integer, list of words, etc.)
def _extract_answer_bbh(text: str, task: Task) -> str:

    task_regex = None
    match task.task_type:
        case TaskType.MULTIPLE_CHOICE:
            task_regex = r"\(?(?P<answer>[A-Z])\)?(?!\w)"
        case TaskType.BINARY:
            options = task.binary_options  # e.g., ["Yes", "No"]
            task_regex = r"(?P<answer>" + "|".join(options) + r")"
        case TaskType.OPEN_ENDED:
            if task.task_name == "dyck_languages":
                # match closing parentheses separated by spaces
                task_regex = r"(?P<answer>[\)\]\}\>\s]+)"
            elif (
                task.task_name == "multistep_arithmetic_two"
                or task.task_name == "object_counting"
            ):
                # match integers
                task_regex = r"(?P<answer>[-+]?\d+)"
            elif task.task_name == "word_sorting":
                # match list of words separated by spaces or commas (and & - for 'r&d' in question_id 245)
                # word = r"[\w&']"
                task_regex = r"(?P<answer>[\w&']+(?:[,\s]+[\w&']+)+)"

    def enclosed_pattern(pattern: str) -> str:
        # Zero or more of the following enclosers
        bold_markdown = r"\**"
        single_quotes = r"'*"
        backtick = r"`*"
        encloser = f"(?:{bold_markdown}|{single_quotes}|{backtick})"
        return encloser + pattern + encloser

    # Enclose the task regex in bold markdown or quotes
    task_regex = enclosed_pattern(task_regex)

    def match_pattern_result(pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if task.task_name != "word_sorting":
                return match.group('answer').strip()
            else:
                # replace commas, newlines, tabs, carriage returns, etc. with empty strings
                return match.group('answer').replace(',', '').replace('\n', '').replace('\t', '').replace('\r', '').strip()
        return None

    # First try: base_pattern + task_regex
    base_pattern = r"(?:answer is:?|\**Answer:\**|\**A:\**)\s*"
    pattern = base_pattern + task_regex
    res = match_pattern_result(pattern)
    if res:
        return res

    if task.task_name == "word_sorting":
        # Also try to match list of words at the end of the text
        pattern = r"\n" + task_regex + r"$"
        res = match_pattern_result(pattern)
        if res:
            return res

    if task.task_name != "word_sorting":
        start_of_string_pattern = r"^"
        # Fallback: try to match at the start of the string
        fallback_pattern = start_of_string_pattern + task_regex
        res = match_pattern_result(fallback_pattern)
        if res:
            return res

    # Fallback 2: return last match of task_regex in the text
    matches = re.findall(task_regex, text, re.IGNORECASE)
    if matches:
        if task.task_name != "word_sorting":
            return matches[-1].strip()
        else:
            return matches[-1].replace(',', '').replace('\n', '').replace('\t', '').replace('\r', '').strip()
