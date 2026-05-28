# Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings

Code for the ICML 2026 paper *Correct Looks Better: Pairwise Comparisons Reveal
Accuracy Rankings* (Remeli & Hardt). Reproduces the main-body experiments
comparing accuracy-based rankings to LLM-as-a-judge rankings (pairwise
Bradley-Terry vs. direct judge) across five benchmarks.

## Setup

```bash
pip install -e src/
cp .env.public .env   # then fill in API keys
```

## Benchmarks and models

| Paper name      | `--dataset`     | Format                |
|-----------------|-----------------|-----------------------|
| MMLU-Pro        | `mmlu_pro`      | MCQ → freeform        |
| GPQA-Diamond    | `gpqa_diamond`  | MCQ → freeform        |
| SimpleQA        | `simple_qa`     | freeform              |
| GSM8K           | `gsm8k`         | freeform              |
| BBH (multitask) | `bbh`           | mixed (17 sub-tasks)  |

Per-benchmark model lists are in `models/models-<dataset>.txt`. Main-body judges
are `openai/gpt-oss-20b` (primary), `openai/gpt-oss-120b`, and `openai/o3`.

## Reproducing the main-body experiments

Run all commands from the repo root. `ranking.py` and `prompt.py` cache their
outputs under `out/`; re-running with the same configuration loads the cache
instead of re-querying the LLM (see [Caching](#caching)).

For experiments from the paper appendices, see [APPENDIX.md](APPENDIX.md).

### 1. Collect model answers (prerequisite)

```bash
python prompt.py --model-name openai/gpt-oss-20b --dataset mmlu_pro \
    --query-mode freeform --litellm
```

Drop `--litellm` to serve a local model with vLLM. Repeat per model per benchmark.

### 2. §5.1 — Pairwise ranking aligns with accuracy

`ranking.py` selects the scoring method via a positional subcommand: omit it
for accuracy, `Pairwise` for pairwise judge methods, `AM` for direct judge.

```bash
# Accuracy-based (gold) ranking
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro

# Pairwise Bradley-Terry ranking
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro \
    Pairwise --what BradleyTerry \
    --judge-model openai/gpt-oss-20b --client litellm
```

Repeat per benchmark with the matching `models/models-<dataset>.txt`.

Rank-correlation between the two rankings (Spearman's ρ, Kendall's τ,
Pearson's R):

```bash
export PYTHONPATH=scripts
python scripts/rank_similarity.py --dataset mmlu_pro --metric rho \
    --judge openai/gpt-oss-20b --method BradleyTerry
```

Add `--bootstrap` for the 95 % CIs (stored in
`out/bootstrap/<dataset>/<models_id>/`).

### 3. §5.2 — Direct judge baseline

```bash
python ranking.py --models-file models/models-simple-qa.txt --dataset simple_qa \
    AM --judge-model openai/gpt-oss-20b --client litellm
```

`AM` without `--with-ground-truth` is the direct judge baseline (no gold
answer shown). With `--with-ground-truth` the judge grades against the gold
answer — used for the SimpleQA / GSM8K accuracy reference.

BT vs. Direct Judge across judges and benchmarks:

```bash
export PYTHONPATH=scripts
python scripts/judge_comparison/judge_comparison_table.py \
    --datasets mmlu_pro gpqa_diamond simple_qa gsm8k bbh \
    --methods BradleyTerry --metric tau
```

The weak-judge regime on SimpleQA is the same pipeline with judges of varying
strength: `gpt-oss-20b` (weak), `gpt-oss-120b` (middle), `o3` (strong).

### 4. §5.3 — Bias correction

Bradley-Terry with feature controls for *style* (length, formatting) and
*self-preference* (judge and candidate from the same model family):

```bash
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro \
    Pairwise --what BradleyTerry \
    --judge-model openai/gpt-oss-20b --client litellm \
    --control-for style                       # or: self-preference, or both
```

Re-run `scripts/rank_similarity.py` with `--bias style|self-preference|both`
to compare against the uncorrected ranking.

### 5. §5.4 — Echo as a causal driver on non-discriminative pairs

(For the discriminative vs. non-discriminative split itself, see
[APPENDIX.md](APPENDIX.md#appendix-d--discriminative-vs-non-discriminative-pairs).)

Detect echo in collected answers:

```bash
python echo_detection.py --dataset bbh --models-file models/models-bbh.txt
```

The detector is hard-wired to `openai/gpt-oss-120b` via **vLLM** — you need
the weights locally and a GPU with enough memory. The detection prompt lives
in `data/prompts/echo_detection/`.

Run the controlled intervention — add echo to one answer, re-query the judge,
measure the causal effect:

```bash
python scripts/pp_experiments/intervention_study/run_intervention.py \
    --intervention add_echo --dataset bbh \
    --judge openai/o3 --n 500 --seed 42
```

## Repository layout

```
src/rank_no_eval/        installable package (clients, eval objects, queries, rankers)
prompt.py                collect model answers
ranking.py               build a ranking with a chosen score
echo_detection.py        run echo detection on collected answers
scripts/                  analysis scripts (rank correlation, judge comparison, intervention study)
models/                  per-benchmark model lists
out/                     cached answers, matches, bootstrap arrays (not checked in)
```

## Caching

`out/` holds all cached artifacts:

- `answers/<query_mode>/<dataset>/<model>.jsonl` — model answers
- `matches/pairwise_comparisons/<hash>/<dataset>/...` — pairwise judgments
- `matches/answer_matching/<judge>/<dataset>/<model>.csv` — direct-judge labels
- `bootstrap/<dataset>/<models_id>/...` — bootstrap rankings

`<hash>` is an MD5 of the sorted model-ID set, tracked in
`out/model_set_ids.txt`. Rankings are recomputed from cached matches on each
invocation — delete the relevant cache file to force re-collection.

## Citation

```bibtex
@inproceedings{remeli2026correct,
  title     = {Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings},
  author    = {Remeli, Mina and Hardt, Moritz},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
