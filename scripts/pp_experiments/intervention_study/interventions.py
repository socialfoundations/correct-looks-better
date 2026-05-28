"""
Echo intervention functions and sample construction.

Given a pairwise sample with diff_echo=0 (either neither or both answers echo),
produces intervention records for all three echo conditions (-1, 0, +1).

Two sub-cases of diff_echo=0:

  echo_a=0, echo_b=0  (neither echoes)
    echo= 0 : original                    → (A,       B)       [cached]
    echo=+1 : add echo to A               → (A_echo,  B)
    echo=-1 : add echo to B               → (A,       B_echo)

  echo_a=1, echo_b=1  (both echo)
    echo= 0 : remove echo from both       → (A_clean, B_clean)
    echo=+1 : keep A, remove echo from B  → (A,       B_clean)
    echo=-1 : remove echo from A, keep B  → (A_clean, B)

Add echo: append (question + answer) × N to the answer.
Remove echo: LLM extraction call — see remove_echo().
"""

import json as _json
from typing import List

from config import N_ECHO_REPS


# ---------------------------------------------------------------------------
# Core echo manipulation
# ---------------------------------------------------------------------------

def add_echo(question: str, answer: str, n_reps: int) -> str:
    """Append N repetitions of (question + answer) to simulate echo."""
    rep = f"\n\n{question}\n{answer}"
    return answer + rep * n_reps


_ECHO_EXTRACTION_SYSTEM = (
    "An LLM answer contains an echo failure mode: after producing its final "
    "answer, the model continued generating by either (1) repeating the same "
    "phrase or token many times, or (2) generating new question–answer pairs "
    "that follow the same template as the original. "
    "Your task is to extract only the clean final answer, removing all echo "
    'content that appears after it. Return your result as JSON: {"clean_answer": "..."}.'
)


class EchoExtractionQuery:
    """Query method for remove_echo LLM calls (used with VLLMClient)."""

    max_tokens = 2048
    temperature = 0
    oss_reasoning_effort = "Low"
    answer_format = {
        "json_schema": {
            "name": "clean_answer",
            "schema": {
                "type": "object",
                "properties": {"clean_answer": {"type": "string"}},
                "required": ["clean_answer"],
                "additionalProperties": False,
            },
        }
    }

    def get_system_prompt(self, sample: dict) -> str:
        return _ECHO_EXTRACTION_SYSTEM

    def get_prompt(self, sample: dict) -> str:
        return f"Question: {sample['question']}\n\nAnswer (with echo):\n{sample['answer']}"

    @property
    def completion_kwargs(self) -> dict:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


def remove_echo(question: str, answer: str, client) -> str:
    """Extract the final answer, stripping echo content via an LLM call.

    Args:
        question: The original question posed to the model.
        answer: The model's answer, which contains echo content.
        client: A pre-built VLLMClient (or compatible client) initialised with
                EchoExtractionQuery. Creating the client is expensive; callers
                should create it once and reuse it.

    Returns:
        Clean answer with echo removed.

    Raises:
        ValueError: If the LLM response cannot be parsed.
    """
    responses = client.collect_model_answers([{"question": question, "answer": answer}])
    try:
        return _json.loads(responses[0])["clean_answer"]
    except (KeyError, _json.JSONDecodeError) as e:
        raise ValueError(
            f"remove_echo: could not parse response: {responses[0]!r}"
        ) from e


# ---------------------------------------------------------------------------
# Intervention record construction
# ---------------------------------------------------------------------------

def build_interventions(
    question_id: str,
    model_a: str,
    model_b: str,
    question: str,
    answer_a: str,
    answer_b: str,
    echo_a: int,
    echo_b: int,
    starting_echo: int,
    one_correct: int,
    original_preference: int,
    answer_a_clean: str | None = None,
    answer_b_clean: str | None = None,
) -> List[dict]:
    """Build intervention records for all three echo conditions.

    Only diff_echo=0 pairs are accepted (echo_a == echo_b).

    For the neither-echoes sub-case (echo_a=0, echo_b=0):
      - echo=0 uses the original cached preference (no new judge call)
      - echo=+1 and echo=-1 add echo to A or B respectively

    For the both-echo sub-case (echo_a=1, echo_b=1):
      - answer_a_clean and answer_b_clean must be pre-computed via remove_echo()
      - All three conditions require new judge calls

    Args:
        starting_echo: 0 if neither answer echoes (add_echo experiment),
                       1 if both answers echo (remove_echo experiment).
                       Recorded verbatim in each output record.

    Returns a list of dicts, one per condition.
    Fields per record:
      question_id, model_a, model_b, question, starting_echo, one_correct,
      echo_condition, intervention, answer_a, answer_b,
      n_reps (int or None), needs_judge (bool), preference (int or None)
    """
    if echo_a != echo_b:
        raise ValueError(
            f"build_interventions expects diff_echo=0 pairs (echo_a == echo_b), "
            f"got echo_a={echo_a}, echo_b={echo_b}"
        )

    n_reps = N_ECHO_REPS

    base = dict(
        question_id=question_id,
        model_a=model_a,
        model_b=model_b,
        question=question,
        starting_echo=starting_echo,
        one_correct=one_correct,
    )

    records: List[dict] = []

    def add(echo_cond: int, intervention: str, ans_a: str, ans_b: str,
            reps: int | None, needs_judge: bool):
        records.append({
            **base,
            "echo_condition": echo_cond,
            "intervention": intervention,
            "answer_a": ans_a,
            "answer_b": ans_b,
            "n_reps": reps,
            "needs_judge": needs_judge,
            "preference": original_preference if not needs_judge else None,
        })

    if echo_a == 0:
        # Neither echoes: add echo to create counterfactuals; original is cached
        add( 0, "original", answer_a,                           answer_b,                           None,   needs_judge=False)
        add(+1, "add_a",    add_echo(question, answer_a, n_reps), answer_b,                           n_reps, needs_judge=True)
        add(-1, "add_b",    answer_a,                           add_echo(question, answer_b, n_reps), n_reps, needs_judge=True)

    else:
        # Both echo: remove echo to create counterfactuals; all need judge calls
        if answer_a_clean is None or answer_b_clean is None:
            raise ValueError(
                "answer_a_clean and answer_b_clean must be provided for both-echo pairs "
                "(echo_a=1, echo_b=1). Pre-compute them with remove_echo() before calling "
                "build_interventions()."
            )
        add( 0, "remove_both", answer_a_clean, answer_b_clean, None, needs_judge=True)
        add(+1, "remove_b",    answer_a,       answer_b_clean, None, needs_judge=True)
        add(-1, "remove_a",    answer_a_clean, answer_b,       None, needs_judge=True)

    return records
