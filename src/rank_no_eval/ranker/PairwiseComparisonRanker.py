from typing import List, Tuple, Dict
from uncertainties import ufloat
from itertools import combinations
import pandas as pd
import random
import numpy as np
import json

from .base import BaseRanker
from .match_manager import PairwiseMatchManager
from rank_no_eval.eval_object import (
    ModelAnswerTrueSkill,
    ModelTrueSkill,
    ModelWinRate,
    ModelElo,
    ModelBradleyTerry,
)
from rank_no_eval.utils import (
    logger,
    QueryMode,
    ClientType,
    Datasets,
)
from rank_no_eval.client import VLLMClient, LiteLLMClient
from rank_no_eval.query import PairwiseComparisonQueries
from rank_no_eval.model_answer import ModelAnswerStore
from rank_no_eval.io import load_freeform_dataset, load_question_ids
from .match_aggregator import (
    TrueSkillMatchAggregator,
    WinRateMatchAggregator,
    EloMatchAggregator,
    BradleyTerryMatchAggregator,
)
from .match_utils import filter_not_attempted, filter_verifiable, filter_unverifiable


class PairwiseComparisonRanker(BaseRanker):

    def __init__(
        self,
        dataset: str,
        judge_model: str,
        client_type: ClientType,
        eval_objects: List[
            ModelAnswerTrueSkill | ModelTrueSkill | ModelWinRate | ModelElo | ModelBradleyTerry
        ] = None,
        answers_only: bool = False,
        control_for: List = [],
        pair_filter: str = None,
    ):
        self.judge_model = judge_model
        self.client_type = client_type
        self.control_for = control_for
        self.pair_filter = pair_filter

        super().__init__(
            eval_objects=eval_objects,
            match_manager=PairwiseMatchManager(),
            dataset=dataset,
        )

        if len(self.eval_objects) > 0:
            self._set_what()
            self._set_match_aggregator()
            self._control_compatibility()

        self.query_method = PairwiseComparisonQueries(
            variant="qa_qa" if self.what == "answer" else "a_a",
            answers_only=answers_only,
        )

    def _control_compatibility(self):
        if self.control_for != []:
            assert isinstance(
                self.eval_objects[0], ModelBradleyTerry
            ), "Controlling for things like style / model family is only supported for BradleyTerry eval objects."
            logger.info(f"Controlling for: {self.control_for}")

    def _set_what(self):
        if isinstance(self.eval_objects[0], ModelAnswerTrueSkill):
            self.what = "answer"
        elif isinstance(
            self.eval_objects[0], (ModelTrueSkill, ModelWinRate, ModelElo, ModelBradleyTerry)
        ):
            self.what = "model"

    def _set_match_aggregator(self):
        if isinstance(self.eval_objects[0], ModelAnswerTrueSkill) or isinstance(
            self.eval_objects[0], ModelTrueSkill
        ):
            self.aggregator = TrueSkillMatchAggregator(self.eval_objects)
        elif isinstance(self.eval_objects[0], ModelWinRate):
            self.aggregator = WinRateMatchAggregator(self.eval_objects)
        elif isinstance(self.eval_objects[0], ModelElo):
            self.aggregator = EloMatchAggregator(self.eval_objects)
        elif isinstance(self.eval_objects[0], ModelBradleyTerry):
            self.aggregator = BradleyTerryMatchAggregator(self.eval_objects)

    def add_eval_object(self, eval_object):
        super().add_eval_object(eval_object)
        # set what after adding the first eval object
        if self.what is None:
            self._set_what()
        return

    def _model_ids(self) -> List[str]:
        return list(set([o.model_id for o in self.eval_objects]))

    def _question_ids(self) -> List[str]:
        question_ids = load_question_ids(self.dataset)
        return question_ids

    def run_comparisons(
        self, matches: List[Dict], run_async: bool, save_raw: bool
    ) -> List[str]:
        logger.info("Running comparisons...")
        # load client
        if self.client_type == ClientType.VLLM:
            client = VLLMClient(
                model_id=self.judge_model, query_method=self.query_method
            )
            # collect model answers
            answers = client.collect_model_answers(matches)
        elif self.client_type == ClientType.LITELLM:
            client = LiteLLMClient(
                model_id=self.judge_model, query_method=self.query_method
            )
            # collect model answers
            answers = client.collect_model_answers(
                matches, run_async=run_async, save_raw=save_raw
            )
        return answers

    def _load_answer_stores(self, with_metadata: bool = False):
        logger.info("Loading answer stores...")
        # preload dataset questions once
        preloaded_dataset = load_freeform_dataset(dataset=self.dataset)
        answer_stores = {
            m_id: ModelAnswerStore(
                model_id=m_id,
                preloaded_dataset=preloaded_dataset,
                dataset=self.dataset,
                query_mode=QueryMode.FREEFORM,
                with_metadata=with_metadata,
            )
            for m_id in self._model_ids()
        }
        return answer_stores

    def _create_model_pairs(self, n: int) -> List[Dict]:
        # TrueSkill-M logic
        answer_stores = self._load_answer_stores()

        logger.info("Creating pairs...")
        model_pairs = list(combinations(self.eval_objects, 2))
        full_pairs = []
        for q in self._question_ids():
            for p1, p2 in model_pairs:
                full_pairs.append(
                    {
                        "p1": {
                            "id": p1.id,
                            **answer_stores[p1.model_id].get(q).__dict__,
                        },
                        "p2": {
                            "id": p2.id,
                            **answer_stores[p2.model_id].get(q).__dict__,
                        },
                    }
                )
        logger.debug(f"Length of full set of pairs: {len(full_pairs)}")
        if n is None:
            return full_pairs
        return random.sample(full_pairs, n)

    def _create_model_answer_pairs(self, n: int) -> List[Dict]:
        # TrueSkill-MA logic
        answer_stores = self._load_answer_stores()

        logger.info("Creating pairs...")
        pairs = []
        for _ in range(n):
            p1, p2 = random.sample(self.eval_objects, 2)
            p1_answer = answer_stores[p1.model_id].get(p1.question_id)
            p2_answer = answer_stores[p2.model_id].get(p2.question_id)
            pairs.append(
                {
                    "p1": {"id": p1.id, **p1_answer.__dict__},
                    "p2": {"id": p2.id, **p2_answer.__dict__},
                }
            )
        return pairs

    def create_pairs(self, n: int) -> List[Dict]:
        if self.what == "model":
            pairs = self._create_model_pairs(n)
        elif self.what == "answer":
            pairs = self._create_model_answer_pairs(n)

        logger.info(f"Created {len(pairs)} pairs")
        return pairs

    def create_matches(self, n: int, run_async: bool, save_raw: bool) -> List[Dict]:
        # create n_comparisons for each player
        pairs = self.create_pairs(n)

        answers = self.run_comparisons(pairs, run_async, save_raw)
        n_empty = len([a for a in answers if a == ""])
        logger.info(f"{n_empty} / {len(answers)} answers are empty.")

        def safe_load(a):
            if a and a.strip():
                try:
                    result = json.loads(a)
                except json.JSONDecodeError:
                    logger.debug(f"JSONDecodeError for answer: {a}")
                    return None
                winner = result.get("winner", None)
                if winner is not None:
                    return {"winner": winner}
            return None

        # throw out pairs with no answer
        filtered = [(p, safe_load(a)) for p, a in zip(pairs, answers)]
        pairs, winner = zip(*[(p, w) for p, w in filtered if w is not None])
        matches = []
        for p in pairs:
            p1 = p["p1"]
            p2 = p["p2"]
            matches.append(
                {
                    "model1": p1["model_id"],
                    "question1": p1["question_id"],
                    "model2": p2["model_id"],
                    "question2": p2["question_id"],
                }
            )
        matches = [{**m, **w} for m, w in zip(matches, winner)]

        # save matches
        self.match_manager.save_matches(
            matches=pd.DataFrame(matches),
            dataset=self.dataset,
            model_id=self.judge_model,
            models=self._model_ids(),
            what=self.what,
            answers_only=self.query_method.answers_only,
        )

        return matches

    def _get_style_metadata(self) -> Dict:
        answer_stores = self._load_answer_stores(with_metadata=True)

        logger.info("Loading style metadata...")
        style_metadata = {}
        for m in self._model_ids():
            style_metadata[m] = {}
            for q in self._question_ids():
                style_metadata[m][q] = answer_stores[m].get(q).metadata
        return style_metadata

    def get_matches(
        self,
        n: int,
        run_async: bool = False,
        save_raw: bool = False,
        load_more: bool = True,
        load_additional: bool = False,
    ) -> List[Dict]:
        matches = self.match_manager.load_matches(
            dataset=self.dataset,
            model_id=self.judge_model,
            models=self._model_ids(),
            what=self.what,
            answers_only=self.query_method.answers_only,
        )
        logger.info(
            f"Loaded {len(matches) if matches is not None else 0} matches from file."
        )

        # filter out matches where winner is not 1 or 2
        if matches is not None:

            def filter_invalid(matches: List[Dict]) -> List[Dict]:
                def is_invalid(m: Dict) -> bool:
                    try:
                        winner = float(m["winner"])
                        return np.isinf(winner) or int(winner) not in [1, 2]
                    except (ValueError, TypeError):
                        return True

                valid_matches = filter(lambda m: not is_invalid(m), matches)
                valid_matches = list(valid_matches)
                logger.info(
                    f"Filtered out {len(matches) - len(valid_matches)} invalid matches."
                )
                return valid_matches

            matches = filter_invalid(matches)

        if matches is None:
            logger.info(f"Creating {n} matches...")
            matches = self.create_matches(n, run_async, save_raw)
        elif load_more:  # matches is not None and load_more is True
            logger.info(f"Loaded {len(matches)} matches from file.")
            if len(matches) < n:
                n_rest = n - len(matches)
                logger.info(f"Loading additional {n_rest} matches...")
                new_matches = self.create_matches(n_rest, run_async, save_raw)
                matches.extend(new_matches)
        elif load_additional:
            logger.info(f"Loading additional {n} matches...")
            new_matches = self.create_matches(n, run_async, save_raw)
            matches.extend(new_matches)

        if self.dataset == Datasets.SIMPLE_QA:
            matches = filter_not_attempted(matches, self._model_ids(), self.dataset)

        if self.pair_filter == "verifiable":
            matches = filter_verifiable(matches, self._model_ids(), self.dataset)
        elif self.pair_filter == "unverifiable":
            matches = filter_unverifiable(matches, self._model_ids(), self.dataset)

        return matches

    def rank(
        self,
        n: int = None,
        run_async: bool = False,
        save_raw: bool = False,
        load_more: bool = True,
        load_additional: bool = False,
        shuffle: bool = False,
        with_replacement: bool = False,
    ):
        # check if there are any eval objects
        assert len(self.eval_objects) > 0, "No eval objects to rank."

        logger.info(f"Ranking {len(self.eval_objects)} objects...")

        if n is None and self.what == "answer":
            # TrueSkill-MA logic
            # eval objects - model-answers (nm)
            # nm * log2(nm) - minimum number of comparisons to get a good ranking
            n = int(len(self.eval_objects) * np.log2(len(self.eval_objects)))
            logger.info(f"Number of pairwise comparisons set to {n}.")

        matches = self.get_matches(
            n=n,
            run_async=run_async,
            save_raw=save_raw,
            load_more=load_more,
            load_additional=load_additional,
        )
        matches = self.sample_items(
            matches, n, shuffle=shuffle, with_replacement=with_replacement
        )

        logger.info(f"Updating scores with {len(matches)} matches...")
        if self.control_for != []:
            self.aggregator.aggregate(
                matches,
                control_for=self.control_for,
                style_metadata=(
                    self._get_style_metadata() if "style" in self.control_for else None
                ),
                judge_model=self.judge_model,
            )
        else:
            self.aggregator.aggregate(matches)

        # sort eval objects by score
        self.eval_objects = sorted(
            self.eval_objects, key=lambda x: x.score, reverse=True
        )

    def calculate_model_scores(self) -> Dict[str, Dict[str, float]]:
        model_ids = self._model_ids()
        models_eval = []
        # aggregate scores on a Model level
        model_scores = {}
        for model_id in model_ids:
            model_scores[model_id] = []

        for o in self.eval_objects:
            model_scores[o.model_id].append(o.score.rating)

        # calculate average score for each model
        for model_id in model_ids:
            scores = model_scores[model_id]
            arr = np.array([ufloat(s.mu, s.sigma) for s in scores])
            mean_score = arr.mean()

            models_eval.append((model_id, mean_score.n, mean_score.s))

        return models_eval

    def model_ranking(self) -> Tuple[str, float]:
        if self.what == "model":
            # sort eval objects by score
            self.eval_objects = sorted(
                self.eval_objects, key=lambda x: x.score, reverse=True
            )
            ranking = [(ma.model_id, ma.score.score) for ma in self.eval_objects]
        elif self.what == "answer":
            model_scores = self.calculate_model_scores()
            ranking = sorted(model_scores, key=lambda x: x[1], reverse=True)
        return ranking

    def calculate_question_scores(self) -> Dict[str, Dict[str, float]]:
        question_ids = self._question_ids()
        questions_eval = []
        # aggregate scores on a Question level
        question_scores = {}
        for question_id in question_ids:
            question_scores[question_id] = []

        for o in self.eval_objects:
            question_scores[o.question_id].append(o.score.rating)

        # calculate average score for each question
        for question_id in question_ids:
            scores = question_scores[question_id]
            arr = np.array([ufloat(s.mu, s.sigma) for s in scores])
            mean_score = arr.mean()

            questions_eval.append((question_id, mean_score.n, mean_score.s))

        return questions_eval

    def question_ranking(self) -> Tuple[str, float]:
        if self.what == "model":
            raise RuntimeWarning("Question ranking is not supported for TrueSkill-M")
        elif self.what == "answer":
            question_scores = self.calculate_question_scores()
            ranking = sorted(question_scores, key=lambda x: x[1], reverse=True)
        return ranking
