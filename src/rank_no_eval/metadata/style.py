import re

STYLE_COLUMNS = ["token_len", "header_count", "list_count", "bold_count"]
LATEX_COLUMNS = ["math_count"]

# Adopted from https://github.com/lmarena/arena-hard-auto/blob/main/utils/add_markdown_info.py
def _count_markdown_elements(markdown_text, suffix):
    counters = {
        f"header_count{suffix}": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }
    return counters


def _count_latex_math_elements(text, suffix):
    # Matches $...$ and \(...\)
    inline_pattern = r'(?<!\\)(?<!\$)\$(?!\$)(?!\s).*?(?<!\s)(?<!\\)\$(?!\$)|\\\(.*?\\\)'
    # REGEX BREAKDOWN:
    # 1. (?<!\\)(?<!\$)\$(?!\$)(?!\s) -> OPENING: A single '$' not escaped, not double,
    #                                    and NOT followed by whitespace (prevents '$ 2').
    # 2. .*?                          -> CONTENT: Match any character lazily.
    # 3. (?<!\s)(?<!\\)\$(?!\$)       -> CLOSING: A single '$' not escaped, not double,
    #                                    and NOT preceded by whitespace (prevents '2 $').
    # 4. |\\\(.*?\\\)                 -> OR: Match the alternative LaTeX syntax '\( ... \)'.

    # Matches $$...$$ and \[...\]
    display_pattern = r"\$\$.*?\$\$|\\\[.*?\\\]"
    # REGEX BREAKDOWN:
    # 1. \$\$.*?\$\$   -> Matches traditional display math blocks ($$ ... $$).
    # 2. |             -> OR operator to catch the alternative LaTeX syntax.
    # 3. \\\[.*?\\\]   -> Matches the modern LaTeX display math delimiters (\[ ... \]).
    #
    # NOTE: Use re.DOTALL with this pattern so the '.*?' can capture math
    # spanning multiple lines.

    counters = {
        f"math_count{suffix}": {
            
            "inline": len( re.findall(inline_pattern,text)),
            
            "display": len(re.findall(display_pattern, text, re.DOTALL)),
        },
    }
    return counters


def _remove_pattern(answer, pattern):
    blocks = pattern.findall(answer)
    for block in blocks:
        answer = answer.replace(block, "")
    return answer


def get_element_counts(df, column):
    codeblock_pattern = re.compile("```([^`]*)```")
    answers = df[column]
    # Remove code block first
    answers_without_code = answers.map(lambda answer: _remove_pattern(answer, codeblock_pattern))
    results = answers_without_code.map(lambda answer: {
        **_count_markdown_elements(answer, ""),
        **_count_latex_math_elements(answer, ""),
    })

    return results.tolist()


def add_markdown_meta(row, encoder):
    conv_meta = {"token_len": len(encoder.encode(row["answer"], disallowed_special=()))}
    return conv_meta | row["markdown_meta"]
