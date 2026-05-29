# Appendix

Reference details for [README.md](README.md): the layout of `out/` and
the prompt-template files under `data/prompts/`.

## Caching

`out/` holds every cached artifact, keyed by configuration:

| Path                                                    | Contents                                              |
|---------------------------------------------------------|-------------------------------------------------------|
| `answers/<query_mode>/<dataset>/<model>.jsonl`          | Model answers                                         |
| `matches/pairwise_comparisons/<hash>/<dataset>/…`       | Pairwise judgments                                    |
| `matches/answer_matching/<judge>/<dataset>/<model>.csv` | Direct-judge / LLM-grader labels                      |
| `bootstrap/<dataset>/<models_id>/…`                     | Resampled rank-correlation arrays (`--bootstrap`)     |

`<hash>` is an MD5 of the sorted model-ID set, tracked in
`out/model_set_ids.txt`. Re-running with the same configuration loads the
cache instead of re-querying. Delete the relevant cache file to force
re-collection; rankings are recomputed on every invocation.

## Prompt templates

All prompts are Jinja files under `data/prompts/`:

| Use                                                     | Template                                                                                   |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Direct judge (`CORRECT` / `INCORRECT` / `NOT_ATTEMPTED`) | `data/prompts/answer_matching_no_gt/prompt.jinja2`                                         |
| Answer matching with ground truth                       | `data/prompts/answer_matching/prompt.jinja2` (+ `simple_qa.jinja2`, `gsm8k.jinja2` overrides) |
| Pairwise comparison (question + both answers)           | `data/prompts/pairwise_comparison/qa_qa/{system_prompt,prompt}.jinja2`                     |
| Pairwise comparison (answers only)                      | `data/prompts/pairwise_comparison/answers_only/{system_prompt,prompt}.jinja2`              |
| Echo detection                                          | `data/prompts/echo_detection/{system_prompt,prompt}.jinja2`                                |
| Per-dataset answer generation                           | `data/prompts/{MCQ,freeform}/<dataset>/`                                                   |
