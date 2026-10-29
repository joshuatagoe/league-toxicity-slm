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

## Risks

- Only 40 toxic messages in `E707`, so recall is noisy; the intervals and the `FULL` rerun cover this.
- Some "false alarms" may be debatable labels; reading them in step 9 tells which.
- Claude can't run at temperature 0, so results vary slightly between runs.
- The brief also wants a Robustness column: a hand-written set of about 40 pressure messages (sarcasm, quotes, self-criticism, harmless trigger words), added after the go/no-go decision.
