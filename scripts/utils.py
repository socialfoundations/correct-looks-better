import pandas as pd

from rank_no_eval.io import load_question_ids, Datasets
from rank_no_eval.ranker import AccuracyMCQRanker, AnswerMatchRanker,  PairwiseComparisonRanker
from rank_no_eval import (
    ModelAnswerTrueSkill,
    ModelTrueSkill,
    ModelWinRate,
    ModelElo,
    ModelBradleyTerry,
    ModelAccuracy,
    ModelMultiTaskAccuracy,
    ModelAnswerMatch,
    ModelMultiTaskAnswerMatch,
)
from rank_no_eval.utils import ClientType

# methods that use a_a prompting
a_a = ["TrueSkill-MA"]

# methods that use qa_qa prompting
qa_qa = ["TrueSkill-M", "WinRate", "Elo", "BradleyTerry"]

data_configs = {
    "mmlu_pro": {
        "n_pairwise_comps": {
            **{m: 74129 for m in a_a},  # assign 74129 to all methods in a_a
            **{m: 32538 for m in qa_qa},  # assign 32538 to all methods in qa_qa
        },
        "models_id": "79359d33",
        "batch_size": 1000,
        "models_file": "models/models-mmlu.txt",
        "ground_truth": "Accuracy",
    },
    "gpqa_diamond": {
        "n_pairwise_comps": {
            **{m: 10057 for m in a_a},
            **{m: 3528 for m in qa_qa},
        },
        # "models_id": "b6f8a002", # old
        "models_id": "58ef55fe",
        "batch_size": 500,
        # "models_file": "models/models-gpqa-old.txt",
        "models_file": "models/models-gpqa.txt",
        "ground_truth": "Accuracy",
    },
    "mt_bench": {
        "n_pairwise_comps": {
            **{m: 4275 for m in a_a},
            **{m: 1200 for m in qa_qa},
        },
        "models_id": "7b2b31d2",
        "batch_size": 250,
        "models_file": "models/models-mt-bench.txt",
        "ground_truth": "BradleyTerry-human",
    },
    "simple_qa": {
        "n_pairwise_comps": {
            **{m: 39045 for m in a_a},
            **{m: 9778 for m in qa_qa},
        },
        # "models_id": "aeca28db",  # old
        "models_id": "f6d8006f",
        "batch_size": 1000,
        "models_file": "models/models-simple-qa.txt",
        # "models_file": "models/models-simple-qa-old.txt",
        "ground_truth": "Accuracy",
    },
    "arena_hard": {
        "n_pairwise_comps": {
            **{m: 123159 for m in a_a},
            **{m: 84582 for m in qa_qa},
        },
        "models_id": "18c87b12",
        "batch_size": 1000,
        "models_file": "models/models-arena-hard.txt",
        "ground_truth": "BradleyTerry-human",
    },
    "gsm8k": {
        "n_pairwise_comps": {
            **{m: 94743 for m in qa_qa},
        },
        "models_id": "622890cd",
        "batch_size": 1000,
        "models_file": "models/models-gsm8k.txt",
        "ground_truth": "Accuracy",
    },
    "gsm8k-ttt": {
        "n_pairwise_comps": {
            **{m: 95506 for m in qa_qa},
        },
        "models_id": "622890cd",
        "batch_size": 1000,
        "models_file": "models/models-gsm8k.txt",
        "ground_truth": "Accuracy",
    },
    "bbh": {
        "n_pairwise_comps": {
            **{m: 23164 for m in qa_qa},
        },
        "models_id": "167a403d",
        "batch_size": 1000,
        "models_file": "models/models-bbh.txt",
        "ground_truth": "Accuracy",
    },
}

def _dataset_models(dataset):
    with open(data_configs[dataset]["models_file"], "r") as f:
        model_ids = [line.strip() for line in f if line.strip()]
    return model_ids


def load_model_rankings(
    dataset,
    method,
    judge_model,
    shuffle: bool,
    with_replacement: bool,
    n: int = None,
    answers_only: bool = False,
    bias: str | None = None,
):
    model_ids = _dataset_models(dataset)

    question_ids = load_question_ids(Datasets(dataset))
    control_for = []
    if method.startswith("TrueSkill"):
        eval_objects = [ModelTrueSkill(id=mid) for mid in model_ids]
    elif method.startswith("WinRate"):
        eval_objects = [ModelWinRate(id=mid) for mid in model_ids]
    elif method.startswith("BradleyTerry"):
        eval_objects = [ModelBradleyTerry(id=mid) for mid in model_ids]
        if bias == "style":
            control_for.append("style")
        elif bias == "self-preference":
            control_for.append("self-preference")
        elif bias == "both":
            control_for.extend(["style", "self-preference"])
    elif method.startswith("Elo"):
        eval_objects = [ModelElo(id=mid) for mid in model_ids]
    elif method.startswith("TrueSkill-MA"):
        eval_objects = [
            ModelAnswerTrueSkill(model_id=mid, question_id=q_id)
            for q_id in question_ids
            for mid in model_ids
        ]

    ranker = PairwiseComparisonRanker(
        dataset=dataset,
        judge_model=judge_model,
        client_type=ClientType.LITELLM,
        eval_objects=eval_objects,
        answers_only=answers_only,
        control_for=control_for,
    )
    ranker.rank(
        n=n, load_more=False, shuffle=shuffle, with_replacement=with_replacement
    )

    return ranker.model_ranking()  # return full ranking with scores


def load_llm_judge_ranking(dataset, judge_model: str = "openai/gpt-oss-120b", bootstrap: bool = False, n: int = None):
    model_ids = _dataset_models(dataset)

    if Datasets(dataset).obj.is_multitask:
        eval_objects = [
            ModelMultiTaskAnswerMatch(id=model_id, tasks=[t.task_name for t in Datasets(dataset).obj.tasks])
            for model_id in model_ids
        ]
    else:
        eval_objects = [ModelAnswerMatch(id=model_id) for model_id in model_ids]

    
    ranker = AnswerMatchRanker(
        dataset=dataset,
        eval_objects=eval_objects,
        judge_model=judge_model,
        client_type=ClientType.VLLM,
        with_ground_truth=False,
    )
    ranker.rank(bootstrap=bootstrap, n=n)
    return ranker.model_ranking()


def load_accuracy_ranking(dataset, bootstrap: bool = False, n: int = None):
    model_ids = _dataset_models(dataset)

    is_multitask = Datasets(dataset).obj.is_multitask
    if dataset in ["simple_qa", "gsm8k"]:
        eval_objects = [ModelAnswerMatch(id=model_id) for model_id in model_ids]
        ranker = AnswerMatchRanker(
            dataset=dataset,
            eval_objects=eval_objects,
            judge_model="openai/gpt-oss-120b",
            client_type=ClientType.VLLM,
            with_ground_truth=True,
        )
    else:
        if is_multitask:
            eval_objects = [
                ModelMultiTaskAccuracy(id=model_id, tasks=[t.task_name for t in Datasets(dataset).obj.tasks])
                for model_id in model_ids
            ]
        else:
            eval_objects = [ModelAccuracy(id=model_id) for model_id in model_ids]   
        ranker = AccuracyMCQRanker(
            dataset=dataset,
            eval_objects=eval_objects,
        )
    ranker.rank(bootstrap=bootstrap, n=n)
    return ranker.model_ranking()


def load_ground_truth_ranking(dataset, ground_truth: str = "Accuracy"):
    from rank_no_eval.config import RANKING_OUTPUT_DIR

    models_id = data_configs[dataset]["models_id"]

    df = pd.read_csv(RANKING_OUTPUT_DIR / models_id / dataset / f"{ground_truth}.csv")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    data = df.to_dict('list')

    return list(zip(data["model_id"], data["score"]))


def rank_similarity(score, gt_score, method="tau"):
    assert method in ["tau", "rho", "r"], f"Unknown similarity method: {method}"
    if method == "tau":
        from scipy.stats import kendalltau

        return kendalltau(score, gt_score)
    elif method == "rho":
        from scipy.stats import spearmanr

        return spearmanr(score, gt_score)
    elif method == "r":
        from scipy.stats import pearsonr
        
        return pearsonr(score, gt_score)
