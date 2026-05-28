from typing import List, Dict
import numpy as np
import pandas as pd

from rank_no_eval.utils import logger
from rank_no_eval.utils import Datasets
from .match_manager import AnswerMatchManager, ExactMatchManager
from .match_aggregator import JudgeAnswers, letter_to_answer


def answer_metadata_accuracy(models_list: List[str], dataset: Datasets) -> Dict[str, float]:
    not_attempted_qids = {}
    correct_qids, incorrect_qids = {}, {}
    em = ExactMatchManager()
    for model in models_list:
        correct_qids[model], incorrect_qids[model] = [], []
        if dataset == Datasets.BBH:
            matches = []
            for task in Datasets(dataset).obj.tasks:
                task_matches = em.load_matches(dataset=dataset, model_id=model, task=task)
                if task_matches is not None:
                    matches.append(task_matches)
            matches = pd.concat(matches, ignore_index=True) if matches else None
        else:
            matches = em.load_matches(dataset=dataset, model_id=model)
        if matches is not None:
            na_matches = matches[
                matches.apply(
                    lambda x: x["extracted_answer"] is None or x["extracted_answer"] == "", axis=1 # empty string or None
                )
            ]
            attempted_matches = matches.drop(na_matches.index)
            c_matches = attempted_matches[
                attempted_matches.apply(
                    lambda x: x["extracted_answer"].lower() == x["answer"].lower(),
                    axis=1,
                )
            ]
            ic_matches = attempted_matches[
                attempted_matches.apply(
                    lambda x: x["extracted_answer"].lower() != x["answer"].lower(),
                    axis=1,
                )
            ]
            not_attempted_qids[model] = list(na_matches["question_id"].unique())
            correct_qids[model] = list(c_matches["question_id"].unique())
            incorrect_qids[model] = list(ic_matches["question_id"].unique())
            logger.debug(
                f"Model {model}: {len(not_attempted_qids[model])} not attempted questions, "
                f"{len(correct_qids[model])} correct answers, "
                f"{len(incorrect_qids[model])} incorrect answers."
            )
    return not_attempted_qids, correct_qids, incorrect_qids


def answer_metadata_AM_judge(models_list: List[str], dataset: Datasets) -> Dict[str, List[str]]:
    not_attempted_qids = {}
    correct_qids, incorrect_qids = {}, {}
    mm = AnswerMatchManager(judge_id="openai/gpt-oss-120b")
    for model in models_list:
        not_attempted_qids[model] = []
        correct_qids[model], incorrect_qids[model] = [], []
        matches = mm.load_matches(dataset=dataset, model_id=model)
        if matches is not None:
            # matches where judge says answer is NOT ATTEMPTED
            n_a_matches = matches[
                matches.apply(
                    lambda x: letter_to_answer.get(x["judge_answer"], None)
                    == JudgeAnswers.NOT_ATTEMPTED,
                    axis=1,
                )
            ]
            c_matches = matches[
                matches.apply(
                    lambda x: letter_to_answer.get(x["judge_answer"], None)
                    == JudgeAnswers.CORRECT,
                    axis=1,
                )
            ]
            ic_matches = matches[
                matches.apply(
                    lambda x: letter_to_answer.get(x["judge_answer"], None)
                    == JudgeAnswers.INCORRECT,
                    axis=1,
                )
            ]

            not_attempted_qids[model] = list(n_a_matches["question_id"].unique())
            correct_qids[model] = list(c_matches["question_id"].unique())
            incorrect_qids[model] = list(ic_matches["question_id"].unique())
            logger.debug(
                f"Model {model}: {len(not_attempted_qids[model])} not attempted questions, "
                f"{len(correct_qids[model])} correct answers, "
                f"{len(incorrect_qids[model])} incorrect answers."
            )
    return not_attempted_qids, correct_qids, incorrect_qids


def _load_correctness_metadata(models_list: List[str], dataset: Datasets):
    if dataset in [Datasets.SIMPLE_QA, Datasets.GSM8K]:
        return answer_metadata_AM_judge(models_list, dataset)
    else:
        return answer_metadata_accuracy(models_list, dataset)


def filter_not_attempted(matches: List[Dict], models_list: List[str], dataset: Datasets) -> List[Dict]:
    logger.debug(f"Filtering out not attempted questions from {len(matches)} matches...")
    not_attempted_qids, _, _ = _load_correctness_metadata(models_list, dataset)
    logger.debug(f"Not attempted qids from {models_list[0]}: {not_attempted_qids[models_list[0]]}")

    def both_attempted(match: Dict) -> bool:
        m1, m2 = match["model1"], match["model2"]
        q1, q2 = str(match["question1"]), str(match["question2"])
        return q1 not in not_attempted_qids[m1] and q2 not in not_attempted_qids[m2]

    filtered_matches = list(filter(both_attempted, matches))
    logger.debug(f"Number of matches after filtering: {len(filtered_matches)}")
    return filtered_matches


def filter_verifiable(matches: List[Dict], models_list: List[str], dataset: Datasets) -> List[Dict]:
    """Keep only pairs where one model is correct and the other is incorrect."""
    logger.debug(f"Filtering for verifiable pairs from {len(matches)} matches...")
    _, correct_qids, incorrect_qids = _load_correctness_metadata(models_list, dataset)

    def one_correct_one_incorrect(match: Dict) -> bool:
        m1, m2 = match["model1"], match["model2"]
        q1, q2 = str(match["question1"]), str(match["question2"])
        return (q1 in correct_qids[m1] and q2 in incorrect_qids[m2]) or (q1 in incorrect_qids[m1] and q2 in correct_qids[m2])

    filtered_matches = list(filter(one_correct_one_incorrect, matches))
    logger.debug(f"Number of verifiable matches: {len(filtered_matches)}")
    return filtered_matches


def filter_unverifiable(matches: List[Dict], models_list: List[str], dataset: Datasets) -> List[Dict]:
    """Keep only pairs where both models are correct or both are incorrect."""
    logger.debug(f"Filtering for unverifiable pairs from {len(matches)} matches...")
    _, correct_qids, incorrect_qids = _load_correctness_metadata(models_list, dataset)

    def both_correct(match: Dict) -> bool:
        m1, m2 = match["model1"], match["model2"]
        q1, q2 = str(match["question1"]), str(match["question2"])
        return q1 in correct_qids[m1] and q2 in correct_qids[m2]

    def both_incorrect(match: Dict) -> bool:
        m1, m2 = match["model1"], match["model2"]
        q1, q2 = str(match["question1"]), str(match["question2"])
        return q1 in incorrect_qids[m1] and q2 in incorrect_qids[m2]

    filtered_matches = list(filter(lambda m: both_correct(m) or both_incorrect(m), matches))
    logger.debug(f"Number of unverifiable matches: {len(filtered_matches)}")
    return filtered_matches


def win_rate_matrix(matches: List[Dict], models: List[str]) -> np.array:
    n = len(models)
    wins_matrix = np.array([[0] * n for _ in range(n)], dtype=np.float32)
    total_matches_matrix = np.array([[0] * n for _ in range(n)], dtype=np.float32)
    for m in matches:
        total_matches_matrix[models.index(m["model1"])][models.index(m["model2"])] += 1
        total_matches_matrix[models.index(m["model2"])][models.index(m["model1"])] += 1
        if m["winner"] == 1:
            wins_matrix[models.index(m["model1"])][models.index(m["model2"])] += 1
        elif m["winner"] == 2:
            wins_matrix[models.index(m["model2"])][models.index(m["model1"])] += 1
    logger.debug("Wins matrix:\n%s", wins_matrix)
    logger.debug("Total matches matrix:\n%s", total_matches_matrix)

    # avoid division by 0
    total_matches_matrix[total_matches_matrix == 0] = 1
    # divide by total matches and avoid division by 0
    wins_matrix = wins_matrix / total_matches_matrix

    logger.debug("Win rates:\n%s", wins_matrix)

    return wins_matrix
