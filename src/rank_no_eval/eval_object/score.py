from __future__ import annotations

from typing import Tuple
from trueskill import Rating, rate_1vs1

from .base import BaseScore


class AccuracyScore(BaseScore):
    def __init__(self):
        self.n_correct: int = 0
        self.n_invalid: int = 0
        self.n_empty: int = 0
        self.n_total: int = 0

    @property
    def score(self) -> float:
        """Calculate accuracy as the ratio of correct answers to total answers."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    def exact_match(self, model_answer: str, ground_truth: str):
        if model_answer is None:
            self.n_invalid += 1
        elif model_answer == "":
            self.n_empty += 1
        else:
            if model_answer.lower() == ground_truth.lower():
                self.n_correct += 1
        self.n_total += 1

    def update(self, match: Tuple[str, str]):
        model_answer, ground_truth = match
        self.exact_match(model_answer=model_answer, ground_truth=ground_truth)

    def __repr__(self):
        return f"AccuracyScore(score={self.score*100:.2f}%, n_correct={self.n_correct}, n_empty={self.n_empty}, n_invalid={self.n_invalid}, n_total={self.n_total})"


class AnswerMatchScore(BaseScore):
    def __init__(self):
        self.n_correct: int = 0
        self.n_not_attempted: int = 0
        self.n_invalid: int = 0
        self.n_total: int = 0

    @property
    def score(self) -> float:
        """Calculate accuracy as the ratio of correct answers to total answers."""
        if self.n_total == 0:
            return 0.0
        # return self.n_correct / self.n_total
        total_valid = self.n_total - self.n_not_attempted - self.n_invalid
        if total_valid == 0:
            return 0.0
        return self.n_correct / total_valid

    def update(
        self,
        match: Tuple[str, str],
        is_correct: bool,
        is_not_attempted: bool,
        is_invalid: bool = False,
    ):
        model_answer, ground_truth = match
        if is_correct:
            self.n_correct += 1
        if is_not_attempted:
            self.n_not_attempted += 1
        if is_invalid:
            self.n_invalid += 1
        self.n_total += 1

    def __repr__(self):
        return f"JudgeScore(score={self.score*100:.2f}%, n_correct={self.n_correct}, n_not_attempted={self.n_not_attempted}, n_invalid={self.n_invalid}, n_total={self.n_total})"


class WinRateScore(BaseScore):
    def __init__(self):
        self.n_wins: int = 0
        self.n_total: int = 0

    @property
    def score(self) -> float:
        """Calculate accuracy as the ratio of correct answers to total answers."""
        if self.n_total == 0:
            return 0.0
        return self.n_wins / self.n_total

    def register(self, win: bool):
        self.n_total += 1
        if win:
            self.n_wins += 1

    def update(self, match: Tuple[str, str], is_winner: bool):
        self.register(is_winner)

    def __repr__(self):
        return f"WinRateScore(score={self.score*100:.2f}%, n_wins={self.n_wins}, n_total={self.n_total})"


class EloScore(BaseScore):
    def __init__(self, k: int = 32, initial_rating: float = 1000.0):
        self.rating: float = initial_rating
        self.k = k

    @property
    def score(self) -> float:
        return self.rating

    def update(self, match: Tuple[EloScore, EloScore], is_winner: bool):
        p1, p2 = match  # p1=winner score, p2=loser score
        e_winner = 1 / (1 + 10 ** ((p2.rating - p1.rating) / 400))
        e_loser = 1 - e_winner
        if is_winner:
            self.rating = p1.rating + self.k * (1 - e_winner)
        else:
            self.rating = p2.rating + self.k * (0 - e_loser)

    def __repr__(self):
        return f"EloScore(rating={self.rating:.2f})"


class BradleyTerryScore(BaseScore):
    def __init__(self):
        self._score = None

    @property
    def score(self) -> float:
        if self._score is None:
            raise ValueError("BradleyTerry score is not set!")
        return self._score

    @score.setter
    def score(self, score: float):
        self._score = score

    def update(self, match):
        raise NotImplementedError(
            "BradleyTerryScore is just a wrapper class and does not support update() method."
        )

    def __repr__(self):
        return f"BradleyTerryScore(score={self.score})"


class TrueSkillScore(BaseScore):
    def __init__(self):
        self.rating: Rating = Rating()

    @property
    def score(self) -> float:
        return self.rating.mu - 3 * self.rating.sigma

    def update(self, match: Tuple[TrueSkillScore, TrueSkillScore], is_winner: bool):
        p1, p2 = match
        w_new, l_new = rate_1vs1(p1.rating, p2.rating)
        if is_winner:
            self.rating = w_new
        else:
            self.rating = l_new

    def __repr__(self):
        return f"TrueSkillScore(score={self.score:.2f}, mu={self.rating.mu:.2f}, sigma={self.rating.sigma:.2f})"


class MultiTaskScore(BaseScore):
    """Container that holds multiple scores that are mapped to a task. Update is an aggregate of all task scores (mean by default).

    - Backwards compatible: `MultiTaskScore` itself implements `score` (mean of tasks by default)
      so code comparing `.score` still works.
    - Allows adding per-task scores via `add_task()` and updating via update(task), which updates only the specified task.
    """

    def __init__(self, average: str = "macro"):
        assert average in ["macro", "micro"], "Supported averaging strategies are 'macro' and 'micro'"
        self.task_scores: dict[str, BaseScore] = {}
        self.average = average

    def add_task(self, task: str, score: BaseScore):
        self.task_scores[task] = score

    def get_task(self, task: str) -> BaseScore:
        return self.task_scores[task]

    def tasks(self) -> list[str]:
        return list(self.task_scores.keys())

    @property
    def scores(self) -> list[float]:
        return [s.score for s in self.task_scores.values()]

    @property
    def score(self) -> float:
        "Mean score across all tasks."
        if not self.task_scores:
            return 0.0
        if self.average == "macro":
            scores = [s.score for s in self.task_scores.values()]
            return sum(scores) / len(scores)
        elif self.average == "micro": # Supported for AccuracyScore and AnswerMatchScore
            return sum([s.n_correct for s in self.task_scores.values()]) / sum(
                [s.n_total for s in self.task_scores.values()]
            )

    def update_task_score(self, task: str, match, *args, **kwargs):
        if task not in self.task_scores:
            raise KeyError(f"Unknown task '{task}'")
        self.task_scores[task].update(match, *args, **kwargs)

    def update(self, match, task: str, *args, **kwargs):
        """
        Update task specific score.
        """
        self.update_task_score(task, match, *args, **kwargs)

    def __repr__(self):
        return (
            f"MultiTaskScore(n_tasks={len(self.task_scores)}, score={self.score:.2f})"
        )

    def per_task_repr(self) -> str:
        reprs = [f"  - {task}: {score}" for task, score in self.task_scores.items()]
        return "MultiTaskScore:\n" + "\n".join(reprs)
