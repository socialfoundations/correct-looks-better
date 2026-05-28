from __future__ import annotations

from typing import Tuple

from .base import BaseEvalObject, BaseScore
from .score import TrueSkillScore


class ModelAnswerEval(BaseEvalObject):
    def __init__(self, model_id: str, question_id: str, score: BaseScore):
        self._model_id = model_id
        self._question_id = question_id
        super().__init__(id=f"{model_id}-{question_id}", score=score)

    @property
    def model_id(self):
        return self._model_id

    @property
    def question_id(self):
        return self._question_id

    def __repr__(self):
        return f"ModelAnswer(id={self.id}, score={self.score})"


class ModelAnswerTrueSkill(ModelAnswerEval):
    def __init__(self, model_id: str, question_id: str):
        super().__init__(model_id, question_id, score=TrueSkillScore())

    def update(self, match: Tuple[ModelAnswerTrueSkill, ModelAnswerTrueSkill]):
        winner, loser = match
        is_winner = winner.id == self.id
        self.score.update((winner.score, loser.score), is_winner)
