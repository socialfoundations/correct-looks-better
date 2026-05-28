# Appendix experiments

Companion to [README.md](README.md), covering the experiments reported in the
paper's appendices. Same setup and conventions apply (run from repo root,
`pip install -e src/`, `.env` populated).

## Appendix A — Filtering ablation

Not included in this release. The filtering-sensitivity analysis lives on a
separate experimental branch.

## Appendix B — Aggregation method comparison

Rank correlation between accuracy and each pairwise aggregation method
(WinRate, Bradley-Terry, Elo, TrueSkill) for a fixed judge:

```bash
export PYTHONPATH=scripts
python scripts/aggregation_comparison/aggregation_comparison_table.py \
    --judge openai/gpt-oss-20b --metric tau
```

`--latex` / `--markdown` for formatted output. Results discussion in
`scripts/aggregation_comparison/NOTES.md`.

## Appendix C — Extra judges

The main pipeline supports any LiteLLM-routable model — just swap
`--judge-model`. For the appendix judges (`microsoft/phi-4`,
`google/gemma-3-27b-it`):

```bash
python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro \
    Pairwise --what BradleyTerry \
    --judge-model microsoft/phi-4 --client litellm

python ranking.py --models-file models/models-mmlu.txt --dataset mmlu_pro \
    AM --judge-model microsoft/phi-4 --client litellm
```

Then compare against the gold ranking with
`scripts/judge_comparison/judge_comparison_table.py` (extend the script's
`JUDGES` constant if needed).

Per-judge accuracy on each benchmark (used in the weak-vs-strong judge
discussion) is in `out/<judge>-eval.csv`.

## Appendix D — Discriminative vs. non-discriminative pairs

Re-run the BT ranking restricted to pairs where exactly one answer is correct
(`verifiable`) or both are correct/incorrect (`unverifiable`):

```bash
python ranking.py --models-file models/models-bbh.txt --dataset bbh \
    Pairwise --what BradleyTerry \
    --judge-model openai/gpt-oss-20b --client litellm \
    --pair-filter verifiable                  # or: unverifiable
```

Then compare against accuracy with `scripts/rank_similarity.py` as in
[README §2](README.md#2-§51--pairwise-ranking-aligns-with-accuracy).

## Appendix E — Echo as a causal driver

Echo detection and the controlled intervention are covered in
[README §5](README.md#5-§54--echo-as-a-causal-driver-on-non-discriminative-pairs).

## Appendix F — Prompts

All prompt templates are Jinja files under `data/prompts/`; the loader
classes in `src/rank_no_eval/query/` just render them.

| Use | Template |
|---|---|
| Direct judge ("CORRECT / INCORRECT / NOT_ATTEMPTED") | `data/prompts/answer_matching_no_gt/prompt.jinja2` |
| Answer matching with ground truth | `data/prompts/answer_matching/prompt.jinja2` (+ `simple_qa.jinja2`, `gsm8k.jinja2` overrides) |
| Pairwise comparison (question + both answers) | `data/prompts/pairwise_comparison/qa_qa/{system_prompt,prompt}.jinja2` |
| Pairwise comparison (answers only) | `data/prompts/pairwise_comparison/answers_only/{system_prompt,prompt}.jinja2` |
| Echo detection | `data/prompts/echo_detection/{system_prompt,prompt}.jinja2` |
| Per-dataset answer generation | `data/prompts/{MCQ,freeform}/<dataset>/` |
