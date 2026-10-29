# Plan: the prompt-ceiling gate

Approved 2026-10-29.

## Goal

Measure precision, recall and F1 on real League chat for a current OpenAI model and a current Claude model, with three prompt styles each. Compare with the paper's GPT-4o numbers (precision 0.35–0.37, F1 0.47–0.51) and decide go or no-go against the bar in the README. The same run is the brief's Prompt-Ceiling Ablation.

## Confirmed facts

- Test split: `model_creation/1_dataset/2_16000_chatlogs_english_only_test_split.csv`; train split: the matching `_train_split.csv`. Columns include `message` and `label` (0/1), plus `chatlog_id`, `time` and `champion_name`; no usernames.
- The paper's ChatGPT script (`model_creation/2_model_evaluation/model_evaluation_chatgpt_toxic.py`) used the first N rows of the test split, with no random sampling, `gpt-4o`, temperature 0, and prompt P1. It dropped answers it could not parse; this project counts them instead.
- Claude: current models are `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-haiku-5-5`. Claude requests can't send `temperature` (rejected since 2026-10-21). Opus 5.5 can't disable thinking, so runs use effort `low`.

## Assumptions

- OpenAI and Anthropic API keys are available (Week 7 teacher costs are covered).
- The current OpenAI model ID is confirmed from the model list at run time.

## The grid

| Prompt style | What it is |
| --- | --- |
| Zero-shot | The paper's P1 prompt, verbatim. Also the like-for-like replication. |
| Few-shot | The annotators' definition of toxic, plus 12 labelled examples from the train split (6 toxic, 6 not, including sarcasm and slang). |
| Structured | The definition, explicit rules for sarcasm, banter, quotes and self-criticism, then a short reason and the label as JSON. |

2 models × 3 styles on the paper's 707 messages (667 non-toxic, 40 toxic).

## Steps

1. Write the bar and decision rule into the README before any run.
2. Download the train and test CSVs at the pinned commit into `data/raw/`. Check: test = 3,035 rows (2,755 / 280); first 707 rows = 667 / 40. Stop if the counts differ.
3. Freeze the eval sets: `E707` = the first 707 test rows, `FULL` = all 3,035. Hash both. Test rows are never used as examples or training data.
4. Write `prompts/zero_shot_p1.txt`, `few_shot.txt` and `structured.txt`. Few-shot examples come from the train split with a fixed seed.
5. Build `src/providers.py` (one interface for OpenAI and Anthropic, with retries and parallel calls) and `src/run_ceiling.py --model --strategy --eval-set --limit`. Each run writes one JSONL line per message (id, message, true label, raw output, predicted label, parse ok, stop reason, tokens, latency) and a `metrics.json`. Claude runs use structured output for the label and no model fallbacks; refusals are recorded, not skipped.
6. Smoke run: 20 messages × 6 cells. Read every output, confirm parsing, measure tokens and price the full run. The cost is approved before step 7.
7. Full grid on `E707`. Optional calibration: if `gpt-4o` is still served, run zero-shot P1 on it and check the harness lands near the paper's 0.35 / 0.92.
8. `src/report.py` builds the results table from the run files: precision, recall, F1 with 95% bootstrap intervals, refusal and parse-failure counts, latency, cost per million messages, the paper's GPT-4o rows alongside, and every false alarm and miss from the best cell.
9. Decide. If the best cell is within its interval of the bar, rerun that cell on `FULL` (280 toxic). Read the false alarms and write the paragraph naming the failure that survives the best prompt.

## Validation

- Count checks in step 2.
- Baselines on `E707`: "always non-toxic" must score about 0.943 accuracy and 0 recall; a keyword rule is scored too.
- Calibration against the paper's gpt-4o numbers (step 7).
- For every cell, parsed + refused + failed = 707.

## Deviations found while building (2026-10-29)

- **The paper's GPT-4o subset can't be rebuilt.** The first 707 test rows hold 643 non-toxic / 64 toxic, not 667 / 40, and the file's `old_label` column gives 74 toxic, so neither matches. The full test split does match the paper (2,755 / 280). Comparison with the paper's GPT-4o rows is therefore approximate.
- **The benchmark is `E1000`, the first 1,000 test rows** (898 non-toxic / 102 toxic, 10.2%), chosen by the user in place of `E707`. It replaces `E707` everywhere; `FULL` stays for confirming the best cell.
- **Models: `gpt-5.6-sol` and `claude-opus-5-5`,** the top tier of each family callable on this account (listed with `scripts/list_models.py`). GPT-5.6 Sol is newer than the GPT-5.5 assumed in the plan.
- **Smoke run (20 messages × 6 cells):** 120 of 120 calls succeeded with no refusals, parse failures or errors. Measured tokens put the full `E1000` grid at about $13, using Sol's $5/$30 list price as an upper estimate.
- **Same reply format and parser for every model.** No provider-specific structured-output feature is used: every model gets the same prompt text and the same parser (`src/labels.py`), so the comparison doesn't depend on API features.
- **Effort is `low` by default** for models that take one, and is recorded per run. If the best cell lands near the bar, it is rerun at a higher effort before deciding.
- **First baseline result:** the keyword rule scores F1 0.558 on `E1000` (precision 0.547, recall 0.569), already above the paper's GPT-4o (F1 0.507).

- **4-bit evaluation broke the untrained model (found 2026-10-29).** Scored with a 4-bit (NF4) base, untrained Qwen3-1.7B often echoed a word from the message ("gg" → "gg") and failed to answer 933 of 1,000 times. The same model in full precision answered all six probe messages correctly. Evaluation now loads the base in bf16 for every model (`eval.py --four-bit` keeps the old behaviour); the 4-bit runs are kept as `*-nf4` for the record. Training still uses the 4-bit base (QLoRA).

## Decision (2026-10-29)

Go. No cell met the bar; the best (Claude Opus 5.5, zero-shot P1) reached precision 0.594 and F1 0.666 on `FULL`. The step-9 rerun on `FULL` was needed because the cell's `E1000` F1 interval (0.62–0.76) touched 0.75; on `FULL` the interval is 0.62–0.70. The higher-effort rerun noted above was not run, since precision on `FULL` (top of interval 0.64) is far from 0.80. See [CEILING_REPORT.md](CEILING_REPORT.md).

## Risks

- Only 40 toxic messages in `E707`, so recall is noisy; the intervals and the `FULL` rerun cover this.
- Some "false alarms" may be debatable labels; reading them in step 9 tells which.
- Claude can't run at temperature 0, so results vary slightly between runs.
- The brief also wants a Robustness column: a hand-written set of about 40 pressure messages (sarcasm, quotes, self-criticism, harmless trigger words), added after the go/no-go decision.
