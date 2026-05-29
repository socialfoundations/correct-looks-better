# Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings

Code repository for the ICML 2026 paper *Correct Looks Better: Pairwise Comparisons
Reveal Accuracy Rankings* (Remeli & Hardt).

## Install

```bash
pip install -e src/
cp .env.public .env   # then fill in API keys
```

## Benchmarks

| `--dataset`    | Paper name      | Format               | Model list                    |
|----------------|-----------------|----------------------|-------------------------------|
| `mmlu_pro`     | MMLU-Pro        | MCQ → freeform       | `models/models-mmlu.txt`      |
| `gpqa_diamond` | GPQA-Diamond    | MCQ → freeform       | `models/models-gpqa.txt`      |
| `simple_qa`    | SimpleQA        | freeform             | `models/models-simple-qa.txt` |
| `gsm8k`        | GSM8K           | freeform             | `models/models-gsm8k.txt`     |
| `bbh`          | BBH (multitask) | mixed (17 sub-tasks) | `models/models-bbh.txt`       |

All commands are run from the repo root.

## Pipeline

```
prompt.py          ranking.py                 scripts/rank_similarity.py
collect            score & rank by:           compare any two rankings on
answers            • accuracy (GT match)      the same (dataset, model set)
                   • LLM grader (AM, w/ GT)
                   • direct judge   (AM)
                   • pairwise       (Pairwise)
```

## 1. Collect answers

`--query-mode` selects the prompt format:

| Value      | Use for                                                  |
|------------|----------------------------------------------------------|
| `MCQ`      | MMLU-Pro and GPQA-Diamond accuracy ranking (§2a)         |
| `freeform` | Everything else — pairwise comparisons, direct judge, LLM-grader accuracy. On MCQ benchmarks this strips the answer options. |

```bash
# API inference (LiteLLM)
python prompt.py --model-name openai/gpt-oss-20b --dataset mmlu_pro \
    --query-mode freeform --litellm

# Local inference (vLLM); drop --litellm
python prompt.py --model-name Qwen/Qwen3-0.6B --dataset gsm8k \
    --query-mode freeform
```

For MMLU-Pro and GPQA-Diamond, collect **both** modes — `MCQ` for the
accuracy reference, `freeform` for the judge-based rankings. Repeat per
model per benchmark.

## 2. Accuracy ranking

### 2a. Exact match to ground truth (MMLU-Pro, GPQA-Diamond, BBH)

```bash
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro
```

### 2b. LLM grader against the ground truth (SimpleQA, GSM8K)

```bash
python ranking.py --models-file models/models-simple-qa.txt --dataset simple_qa \
    AM --judge-model openai/gpt-oss-20b --client litellm --with-ground-truth
```

## 3. Direct judge ranking

Same `AM` subcommand as §2b but the judge sees only the question and the
answer (no gold):

```bash
python ranking.py --models-file models/models-simple-qa.txt --dataset simple_qa \
    AM --judge-model openai/gpt-oss-20b --client litellm
```

## 4. Pairwise ranking

```bash
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro \
    Pairwise --what BradleyTerry \
    --judge-model openai/gpt-oss-20b --client litellm
```

### `Pairwise` flags

| Flag             | Values                                       | Effect                                                |
|------------------|----------------------------------------------|-------------------------------------------------------|
| `--what`         | `BradleyTerry`, `Elo`, `TrueSkill-M`, `WinRate` | Aggregation method (`BradleyTerry` is the paper's default) |
| `--judge-model`  | any LiteLLM-routable model                   | Judge identity                                        |
| `--control-for`  | `style`, `self-preference`, `both`           | Bias features absorbed by the BT fit                  |
| `--pair-filter`  | `verifiable`, `unverifiable`                 | Restrict to pairs with one / zero or two correct answers (requires §2) |
| `--client`       | `litellm`, `vllm`                            | Judge inference backend                               |

The paper uses `openai/gpt-oss-20b` (weak / primary judge),
`openai/gpt-oss-120b` (middle), and `openai/o3` (strong) for the
weak-judge regime on SimpleQA.

## 5. Compare rankings

```bash
export PYTHONPATH=scripts
```

### 5a. One ranking vs. accuracy

```bash
python scripts/rank_similarity.py --dataset mmlu_pro --metric rho \
    --judge openai/gpt-oss-20b --method BradleyTerry
```

| Flag          | Values                              | Effect                                  |
|---------------|-------------------------------------|-----------------------------------------|
| `--metric`    | `rho`, `tau`, `R`                   | Spearman ρ / Kendall τ / Pearson R      |
| `--method`    | `BradleyTerry`, `AM`, …             | Which §3 / §4 ranking to load           |
| `--bias`      | `style`, `self-preference`, `both`  | Pick a bias-corrected BT ranking (§4)   |
| `--bootstrap` | flag                                | 95 % CIs over bootstrap resamples       |

### 5b. Across judges and benchmarks

```bash
python scripts/judge_comparison/judge_comparison_table.py \
    --datasets mmlu_pro gpqa_diamond simple_qa gsm8k bbh \
    --methods BradleyTerry --metric tau
```

## 6. Echo detection and intervention

Label cached answers for echo (question repetition after the final answer):

```bash
python echo_detection.py --dataset bbh --models-file models/models-bbh.txt
```

Run the controlled intervention (perturb one answer in each pair, re-query
the judge, write paired original / counterfactual verdicts):

```bash
python scripts/pp_experiments/intervention_study/run_intervention.py \
    --intervention add_echo --dataset bbh \
    --judge openai/o3 --n 500 --seed 42
```

| Flag             | Values            | Effect                                 |
|------------------|-------------------|----------------------------------------|
| `--intervention` | `add_echo`        | Perturbation applied to one answer     |
| `--n`            | int               | Number of pairs to sample              |
| `--judge`        | any LiteLLM model | Judge re-queried on the perturbed pair |

## Repository layout

| Path                | Role                                                       |
|---------------------|------------------------------------------------------------|
| `src/rank_no_eval/` | Installable package — clients, queries, rankers            |
| `prompt.py`         | §1 — collect answers                                       |
| `ranking.py`        | §2–§4 — accuracy / direct judge / pairwise rankings        |
| `echo_detection.py` | §6 — label answers for echo                                |
| `scripts/`          | §5 — rank correlation, judge comparison, intervention study |
| `data/prompts/`     | Jinja templates for every judge / generation prompt        |
| `models/`           | Per-benchmark model lists                                  |
| `out/`              | Cached answers, matches, bootstrap arrays (not checked in) |

Cache layout and prompt-template paths are detailed in
[APPENDIX.md](APPENDIX.md).

## Citation

```bibtex
@inproceedings{remeli2026correct,
  title     = {Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings},
  author    = {Remeli, Mina and Hardt, Moritz},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
