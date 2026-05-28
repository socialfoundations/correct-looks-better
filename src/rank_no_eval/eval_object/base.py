import abc
from typing import List, Tuple, Any


class BaseScore(abc.ABC):
    @property
    @abc.abstractmethod
    def score(self) -> float:
        pass

    @abc.abstractmethod
    def update(self, match: Tuple[Any, Any]):
        pass

    def __lt__(self, other):
        return self.score < other.score


class BaseEvalObject(abc.ABC):
    def __init__(self, id: str, score: BaseScore):
        self.id = id
        self.score = score

    def update(self, match: Tuple[Any, Any], *args, **kwargs):
        """Update the score based on a match."""
        self.score.update(match, *args, **kwargs)

    def update_all(self, matches: List[Tuple[Any, Any]]):
        for match in matches:
            self.update(match)
