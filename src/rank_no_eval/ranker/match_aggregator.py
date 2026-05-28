import abc
from typing import List, Tuple, Any, Dict
from enum import Enum
import re
from copy import deepcopy
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

from rank_no_eval.eval_object.base import BaseEvalObject
from rank_no_eval.eval_object import *
from rank_no_eval.utils import logger
from rank_no_eval.metadata import STYLE_COLUMNS, ModelFamily, get_family

# Match Aggregator classes
# in: eval_objects and matches
# out: updated eval_objects


class BaseMatchAggregator(abc.ABC):
    def __init__(self, eval_objects: List[BaseEvalObject]):
        self.eval_objects = eval_objects

    def get_eval_object(self, id: str) -> BaseEvalObject:
        # find eval object with matching id
        for eval_object in self.eval_objects:
            if eval_object.id == id:
                return eval_object
        raise ValueError(f"Eval object with id {id} not found.")

    @abc.abstractmethod
    def aggregate(self, matches: Any):
        pass


class OnlineMatchAggregator(BaseMatchAggregator):
    # class which aggregates matches in an online fashion
    # e.g. order of matches matters because scores are updated after each match

    @abc.abstractmethod
    def process_match(self, match: Tuple[Any, Any]):
        pass

    def aggregate(self, matches: List[Tuple[Any, Any]]):
        for match in matches:
            self.process_match(match)


class OfflineMatchAggregator(BaseMatchAggregator):
    # class which aggregates matches in an offline fashion
    # matches of eval_objects are independent of each other (we can iterate over eval_objects and update their scores separately)
    # OR all eval_objects have to be updated at the same time based on the full set of matches
    def __init__(self, eval_objects: List[BaseEvalObject]):
        super().__init__(eval_objects)


# Answer Matching helper functions
class JudgeAnswers(Enum):
    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    NOT_ATTEMPTED = "Not Attempted"
    INVALID = "Invalid"


letter_to_answer = {
    "A": JudgeAnswers.CORRECT,
    "B": JudgeAnswers.INCORRECT,
    "C": JudgeAnswers.NOT_ATTEMPTED,
}


def parse_judge_answer(judge_answer: str | None) -> str:
    if judge_answer is not None:
        # convert to string if not already
        judge_answer = str(judge_answer).strip()

        # Accept only a single-letter grade or an exact textual label.
        if re.fullmatch(r"[A-C]", judge_answer):
            return letter_to_answer[judge_answer].value

        match = re.fullmatch(
            r"(Correct|Incorrect|Not Attempted)", judge_answer, re.IGNORECASE
        )
        if match:
            return JudgeAnswers(match.group(0).title()).value
    # if no match at all, return Invalid
    return JudgeAnswers.INVALID.value


# Answer Matching Aggregators


class AccuracyMatchAggregator(OfflineMatchAggregator):
    independent = True

    def __init__(self, eval_objects: List[ModelAccuracy]):
        super().__init__(eval_objects)

    def aggregate(self, matches: Dict[str, List], task: str = None):
        for ma in self.eval_objects:
            if task is not None:
                ma.update_all(matches[ma.id], task)
            else:
                ma.update_all(matches[ma.id])


class AnswerMatchAggregator(OfflineMatchAggregator):
    independent = True

    def __init__(self, eval_objects: List[ModelAnswerMatch]):
        super().__init__(eval_objects)

    def aggregate(self, matches: Dict[str, pd.DataFrame], task: str = None):
        for ma in self.eval_objects:
            model_matches = matches[ma.id]
            for _, row in model_matches.iterrows():
                judge_answer = row["judge_answer"]
                parsed_judge_answer = parse_judge_answer(judge_answer)
                def call_update(ma, row, *args, **kwargs):
                    ma.update(
                        match=(row["predicted_answer"], row["answer"]),
                        is_correct=parsed_judge_answer == JudgeAnswers.CORRECT.value,
                        is_not_attempted=parsed_judge_answer
                        == JudgeAnswers.NOT_ATTEMPTED.value,
                        is_invalid=parsed_judge_answer == JudgeAnswers.INVALID.value,
                        *args, **kwargs
                    )
                if task is not None:
                    call_update(ma, row, task_id=task)
                else:
                    call_update(ma, row)


# Pairwise Match helper functions
def eval_obj_id(model_id: str, question_id: str, what: str):
    if what == "answer":
        return f"{model_id}-{question_id}"
    elif what == "model":
        return model_id


def get_winner_loser_id(match: List[Dict], what: str):
    def id(m: str, q: str):
        return eval_obj_id(m, q, what)

    w_idx = int(match["winner"])
    m1, q1 = match["model1"], match["question1"]
    m2, q2 = match["model2"], match["question2"]
    if w_idx == 1:
        w_id, l_id = id(m1, q1), id(m2, q2)
    elif w_idx == 2:
        w_id, l_id = id(m2, q2), id(m1, q1)
    else:
        raise ValueError(f"Invalid winner index: {w_idx}")
    return w_id, l_id


def get_lhs_rhs_id(match: List[Dict], what: str):
    def id(m: str, q: str):
        return eval_obj_id(m, q, what)

    m1, q1 = match["model1"], match["question1"]
    m2, q2 = match["model2"], match["question2"]
    return id(m1, q1), id(m2, q2)


# Pairwise Match Aggregators


class WinRateMatchAggregator(OfflineMatchAggregator):
    independent = True

    def __init__(self, eval_objects: List[ModelWinRate]):
        super().__init__(eval_objects)

    def aggregate(self, matches: List[Tuple[Any, Any]]):
        for match in matches:
            try:
                w_id, l_id = get_winner_loser_id(match, what="model")
                winner, loser = self.get_eval_object(w_id), self.get_eval_object(l_id)
                # run match
                winner.update((winner, loser))
                loser.update((winner, loser))
            except ValueError as e:
                logger.error(f"Error when running match: {e}.\nMatch: {match}")


class TrueSkillMatchAggregator(OnlineMatchAggregator):
    def __init__(self, eval_objects: List[ModelTrueSkill | ModelAnswerTrueSkill]):
        super().__init__(eval_objects)
        self.what = (
            "answer"
            if isinstance(self.eval_objects[0], ModelAnswerTrueSkill)
            else "model"
        )

    def run_match(self, winner: BaseEvalObject, loser: BaseEvalObject):
        # 'freeze' scores for update
        w_copy = deepcopy(winner)
        l_copy = deepcopy(loser)
        # update winner - winner's score gets updated, but w_copy is not
        winner.update((w_copy, l_copy))
        # update loser
        loser.update((w_copy, l_copy))

    def process_match(self, match: List[Dict]):
        try:
            w_id, l_id = get_winner_loser_id(match, what=self.what)
            winner, loser = self.get_eval_object(w_id), self.get_eval_object(l_id)
            # run match
            self.run_match(winner, loser)
        except ValueError as e:
            logger.error(f"Error when running match: {e}.\nMatch: {match}")


class EloMatchAggregator(OnlineMatchAggregator):
    def __init__(self, eval_objects: List[ModelElo]):
        super().__init__(eval_objects)

    def run_match(self, winner: BaseEvalObject, loser: BaseEvalObject):
        w_copy = deepcopy(winner)
        l_copy = deepcopy(loser)
        winner.update((w_copy, l_copy))
        loser.update((w_copy, l_copy))

    def process_match(self, match: List[Dict]):
        try:
            w_id, l_id = get_winner_loser_id(match, what="model")
            winner, loser = self.get_eval_object(w_id), self.get_eval_object(l_id)
            self.run_match(winner, loser)
        except ValueError as e:
            logger.error(f"Error when running match: {e}.\nMatch: {match}")


class BradleyTerryMatchAggregator(OfflineMatchAggregator):
    independent = False  # Here all matches are needed to determine the individual scores of the evaluated objects

    def __init__(self, eval_objects: List[ModelBradleyTerry]):
        super().__init__(eval_objects)

    def _get_style_vector(self, style_metadata: Dict, match: Dict):
        m1, q1 = match["model1"], str(match["question1"])
        m2, q2 = match["model2"], str(match["question2"])

        m1_style = style_metadata[m1][q1]
        m2_style = style_metadata[m2][q2]

        def get_value(element):
            if type(element) is int:
                return element
            elif type(element) is dict:
                return sum(element.values())

        f1 = np.array([get_value(m1_style[col]) for col in STYLE_COLUMNS]).astype(float)
        f2 = np.array([get_value(m2_style[col]) for col in STYLE_COLUMNS]).astype(float)

        style_diff = f1 - f2
        style_sum = f1 + f2

        # divide by style_sum + 1 (add-one smoothing, avoids clipping at ±1)
        style_diff /= (style_sum + 1)

        return style_diff

    def _get_postition_mtx_label(self, matches: List[Dict]):
        P, Y = [], []
        for match in matches:
            try:
                w_idx = int(match["winner"]) if int(match["winner"]) in [1, 2] else 0
            except (ValueError, TypeError):
                w_idx = 0
            l_id, r_id = get_lhs_rhs_id(match, what="model")
            l_idx = self.eval_objects.index(self.get_eval_object(l_id))
            r_idx = self.eval_objects.index(self.get_eval_object(r_id))
            position_vector = [0] * len(self.eval_objects)
            position_vector[l_idx] = 1
            position_vector[r_idx] = -1
            P.append(position_vector)
            Y.append(1 if w_idx == 1 else 0)
        return np.array(P), np.array(Y)

    def _get_style_mtx(self, matches: List[Dict], style_metadata: Dict):
        S = []
        for match in matches:
            S.append(self._get_style_vector(style_metadata, match))
        S = np.array(S)
        # normalize style vectors (per-feature, across all matches)
        S_mean = np.mean(S, axis=0)
        S_std = np.std(S, axis=0)
        S_std[S_std == 0] = 1  # avoid divide-by-zero
        S = (S - S_mean[np.newaxis, :]) / S_std[np.newaxis, :]
        return S

    def _get_model_family_mtx(self, matches: List[Dict]):
        M = []
        for match in matches:
            model_families = [m.value for m in ModelFamily]
            model_family_vector = [0] * len(model_families)
            m1, m2 = match["model1"], match["model2"]
            f1, f2 = get_family(m1), get_family(m2)
            # leave values at 0 if there is a tie between model families
            if f1 != f2:
                model_family_vector[model_families.index(f1.value)] = (
                    1  # lhs model family
                )
                model_family_vector[model_families.index(f2.value)] = (
                    -1
                )  # rhs model family
            M.append(model_family_vector)
        return np.array(M)

    def _get_self_preference_mtx(self, matches: List[Dict], judge_model: str):
        SP = []
        judge_mf = get_family(judge_model)
        logger.debug(f"Judge model family: {judge_mf}")
        for match in matches:
            m1, m2 = match["model1"], match["model2"]
            f1, f2 = get_family(m1), get_family(m2)
            # leave values at 0 if there is a tie between model families
            if f1 != f2:
                SP.append(1 if f1 == judge_mf else -1)
            else:
                SP.append(0)
        return np.array(SP)[:, np.newaxis]

    def aggregate(
        self,
        matches: List[Dict],
        control_for: List = [],
        style_metadata: Dict = None,
        judge_model: str = None,
    ):
        if "style" in control_for:
            assert style_metadata is not None, "style_metadata must be provided"
        if "self-preference" in control_for:
            assert (
                judge_model is not None
            ), "judge_model must be provided for self-preference"

        # filter out tie matches
        matches = [match for match in matches if match["winner"] != "0"]

        X, Y = self._get_postition_mtx_label(matches)

        if "style" in control_for:
            S = self._get_style_mtx(matches, style_metadata)
            logger.debug(f"Style matrix:\n{S}")
            X = np.hstack((X, S))
        if "model_family" in control_for:
            M = self._get_model_family_mtx(matches)
            logger.debug(f"Model family matrix:\n{M}")
            X = np.hstack((X, M))
        if "self-preference" in control_for:
            SP = self._get_self_preference_mtx(matches, judge_model)
            logger.debug(f"Self preference matrix:\n{SP}")
            X = np.hstack((X, SP))

        # Bradley-Terry: logistic regression with no intercept, L2 regularization
        model = LogisticRegression(fit_intercept=False, penalty="l2", C=1.0, solver="lbfgs")
        model.fit(X, Y)

        coefs = model.coef_[0]
        if control_for != []:
            len_style = 0
            if "style" in control_for:
                len_style = len(STYLE_COLUMNS)
                style_coefs = coefs[
                    len(self.eval_objects) : len(self.eval_objects) + len_style
                ]
                logger.debug(f"Style coefficients: {style_coefs}")
            if "model_family" in control_for:
                len_family = len(ModelFamily)
                family_coefs = coefs[
                    len(self.eval_objects)
                    + len_style : len(self.eval_objects)
                    + len_style
                    + len_family
                ]
                logger.debug(f"Model family coefficients: {family_coefs}")
            if "self-preference" in control_for:
                len_self_pref = 1
                self_pref_coefs = coefs[-1:]
                logger.debug(f"Self preference coefficients: {self_pref_coefs}")

        for i, obj in enumerate(self.eval_objects):
            obj.set_BT_score(coefs[i])
