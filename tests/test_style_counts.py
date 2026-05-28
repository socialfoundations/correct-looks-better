import yaml
from pathlib import Path

import pytest

from rank_no_eval.metadata import style


EXAMPLES_PATH = Path(__file__).resolve().parent / "data" / "style_counts.yaml"


def _load_cases(key):
    if not EXAMPLES_PATH.exists():
        return []
    with EXAMPLES_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get(key, [])


MARKDOWN_CASES = _load_cases("markdown")
LATEX_CASES = _load_cases("latex")
DATAFRAME_CASES = _load_cases("dataframe")
MARKDOWN_PARAMS = [(case["text"], case["expected"]) for case in MARKDOWN_CASES]
LATEX_PARAMS = [(case["text"], case["expected"]) for case in LATEX_CASES]


@pytest.mark.parametrize(
    "text, expected",
    MARKDOWN_PARAMS if MARKDOWN_PARAMS else [("", None)],
)
def test_count_markdown_elements(text, expected):
    if not MARKDOWN_CASES:
        pytest.skip("No markdown examples provided")
    result = style._count_markdown_elements(text, "")

    header_counts = (
        result["header_count"]["h1"],
        result["header_count"]["h2"],
        result["header_count"]["h3"],
        result["header_count"]["h4"],
        result["header_count"]["h5"],
        result["header_count"]["h6"],
    )
    expected_header = (
        expected["header"]["h1"],
        expected["header"]["h2"],
        expected["header"]["h3"],
        expected["header"]["h4"],
        expected["header"]["h5"],
        expected["header"]["h6"],
    )
    assert header_counts == expected_header
    assert result["list_count"]["ordered"] == expected["list"]["ordered"]
    assert result["list_count"]["unordered"] == expected["list"]["unordered"]
    assert result["bold_count"]["**"] == expected["bold"]["**"]
    assert result["bold_count"]["__"] == expected["bold"]["__"]


@pytest.mark.parametrize(
    "text, expected",
    LATEX_PARAMS if LATEX_PARAMS else [("", None)],
)
def test_count_latex_math_elements(text, expected):
    if not LATEX_CASES:
        pytest.skip("No latex examples provided")
    result = style._count_latex_math_elements(text, "")
    assert result["math_count"]["inline"] == expected["inline"]
    assert result["math_count"]["display"] == expected["display"]