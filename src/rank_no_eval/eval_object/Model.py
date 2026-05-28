from __future__ import annotations

from typing import Tuple

from .base import BaseEvalObject, BaseScore
from .score import (
    AccuracyScore,
    WinRateScore,
    EloScore,
    TrueSkillScore,
    AnswerMatchScore,
    BradleyTerryScore,
    MultiTaskScore,
)


class ModelEval(BaseEvalObject):
    def __init__(self, id: str, score: BaseScore):
        super().__init__(id, score)

    @property
    def model_id(self):
        return self.id

    def __repr__(self):
        return f"ModelEval(id={self.id}, score={self.score})"


class ModelAccuracy(ModelEval):
    def __init__(self, id: str):
        super().__init__(id, score=AccuracyScore())


class ModelAnswerMatch(ModelEval):
    def __init__(self, id: str):
        super().__init__(id, score=AnswerMatchScore())

    def update(
        self,
        match: Tuple[str, str],
        is_correct: bool,
        is_not_attempted: bool,
        is_invalid: bool,
    ):
        self.score.update(match, is_correct, is_not_attempted, is_invalid)


class ModelWinRate(ModelEval):
    def __init__(self, id: str):
        super().__init__(id, score=WinRateScore())

    def update(self, match: Tuple[str, str]):
        winner, _ = match
        is_winner = winner.id == self.id
        self.score.update(match, is_winner)


class ModelBradleyTerry(ModelEval):
    def __init__(self, id: str):
        super().__init__(id, score=BradleyTerryScore())

    def set_BT_score(self, score: float):
        self.score.score = score

    def update(self, match: Tuple[ModelBradleyTerry, ModelBradleyTerry]):
        raise NotImplementedError(
            "update() method is not supported for the Bradley-Terry model eval object."
        )


class ModelTrueSkill(ModelEval):
    def __init__(self, id: str):
        super().__init__(id, score=TrueSkillScore())

    def update(self, match: Tuple[ModelTrueSkill, ModelTrueSkill]):
        winner, loser = match
        is_winner = winner.id == self.id
        self.score.update((winner.score, loser.score), is_winner)


class ModelElo(ModelEval):
    def __init__(self, id: str, k: int = 32):
        super().__init__(id, score=EloScore(k=k))

    def update(self, match: Tuple[ModelElo, ModelElo]):
        winner, loser = match
        is_winner = winner.id == self.id
        self.score.update((winner.score, loser.score), is_winner)


class ModelMultiTaskAccuracy(ModelEval):
    def __init__(self, id: str, tasks: list[str]):
        super().__init__(id, score=MultiTaskScore(average="micro"))
        for task in tasks:
            self.score.add_task(task, AccuracyScore())

    def update(self, match: Tuple[str, str], task_id: str):
        self.score.update(match, task_id)

    def update_all(self, matches, task_id: str):
        for match in matches:
            self.update(match, task_id)

    def per_task_scores(self) -> str:
        return self.score.per_task_repr()



class ModelMultiTaskAnswerMatch(ModelEval):
    def __init__(self, id: str, tasks: list[str]):
        super().__init__(id, score=MultiTaskScore(average="micro"))
        for task in tasks:
            self.score.add_task(task, AnswerMatchScore())

    def update(self, match: Tuple[str, str], task_id: str, is_correct: bool, is_not_attempted: bool, is_invalid: bool):
        self.score.update(match, task_id, is_correct, is_not_attempted, is_invalid)

    def update_all(self, matches, task_id: str):
        for match, is_correct, is_not_attempted, is_invalid in matches:
            self.update(match, task_id, is_correct, is_not_attempted, is_invalid)

    def per_task_scores(self) -> str:
        return self.score.per_task_repr()