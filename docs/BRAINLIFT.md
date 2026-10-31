# BrainLift: does data beat prompting for League chat toxicity?

**Final submission, 2026-10-30.** It adds a second training seed per size, the paper's own classifier as a baseline, and a heavier dose of the teacher lines (v3) to the early-submission draft of 2026-10-29. The staff held-out set is scored by the graders with `compare.py`, so its result isn't here. Every number here comes from [results/slm_results.md](../results/slm_results.md), [results/ablation1.md](../results/ablation1.md) or [results/relabel_analysis.md](../results/relabel_analysis.md).

## Purpose

Work out when a narrow judgement is better taught with labelled data than asked for in a prompt. The test case is one League of Legends chat line, labelled `toxic` or `nontoxic`, where a false flag punishes an innocent player, so precision matters most.

- **In scope:** single-line classification; precision-first moderation; what a small model learns from expert labels versus teacher-written lines; how much data it takes.
- **Out of scope:** moderation with the surrounding chat or game state; deciding punishments; languages other than English.

## Thesis

A community's standard lives in its labels, not in a description of it. Qwen3-1.7B, trained on about 11,000 expert-labelled League chat lines, holds that standard more precisely than the best frontier model prompted to apply it. Teacher-written lines can add to that standard, but only where the human labels don't push the other way. At 6% of the data and one epoch, they taught "diff" put-downs, which the expert data never mentions. They did not teach sarcasm, which looks like the praise the experts label fine. Four times the dose taught one sarcastic line and cost 0.118 precision, because the model began flagging the sarcasm-shaped chat the experts let pass.

## Did data → behaviour hold?

| Behaviour | Data change | Result | Held? |
| --- | --- | --- | --- |
| Flag real toxic chat precisely | 10,919 expert-labelled rows (v1) | Precision 0.466 → 0.823, a gain of +0.36 (95% CI +0.26 to +0.46). F1 0.468 → 0.718 | **Yes** |
| Meet the bar: precision ≥ 0.80 and F1 ≥ 0.75 | Same | v1 reaches precision 0.823, F1 0.718; the shipped v2 reaches 0.857 and 0.698. Retrained with a second seed, v2 reaches 0.875 and 0.724 | **Half:** the precision part holds, the F1 part doesn't |
| Match a classifier trained on the same expert labels | v2 against the paper's fine-tuned BERT (12,136 expert rows) | Benchmark F1 0.698 vs 0.770 (paired −0.072, CI −0.13 to −0.02), but BERT chose its epoch and threshold on the test split. Pressure set 30/40 vs 25/40 (+5 lines, CI +0.03 to +0.22) | **Split:** BERT leads on the experts' benchmark, v2 on the written rules |
| Catch hostility without insult words | Plus 676 teacher lines aimed at it (v2) | 3 → 4 of 11 toxic pressure lines caught, in both training seeds; the new catch is `jg diff` both times. None of the 3 sarcastic ones | **No**, beyond "diff" put-downs |
| Catch sarcasm with a heavier dose | The same 676 teacher lines 4 times, 19.8% of the data (v3) | 1 of 3 sarcastic pressure lines caught. Precision 0.857 → 0.739 (−0.118, CI −0.19 to −0.05), below the bar; F1 +0.003 | **No:** it trades precision for recall |
| Leave harmless lines with hostile-looking words alone | Same teacher lines | False alarms 5 → 3 of 29 pressure lines with seed 3407, 4 → 4 with seed 42. Pressure set +3 and +1 lines; benchmark F1 −0.021 and +0.034, all within noise | **No**, within seed noise |
| Improve with more of the same data | 5,459 → 10,919 expert rows, two training seeds | F1 +0.010 (CI −0.03 to +0.05) and −0.015 (CI −0.06 to +0.03); pressure score flat | **No gain** past about 5,500 rows, in both seeds |

## Spiky points of view

1. **Writing the standard into the prompt makes a frontier model worse at it.**
   - Claude Opus 5.5 scored precision 0.624 with a one-line prompt.
   - Given the annotators' own guideline, with examples (few-shot) or with a step-by-step format (structured), it fell to 0.351 and 0.346.
   - The guideline's rule ("a message counts as toxic if even a couple of experienced players would find it offensive") reads as "flag when in doubt". The labels show how players actually applied it.
2. **A keyword list is the baseline to beat, and the newest OpenAI model doesn't beat it.** GPT-5.6 Sol's best F1 was 0.568; a keyword rule scored 0.558.
3. **For a judgement with known answers, an LLM judge is a step backwards.** The model answers one word, so a string comparison with the human label grades it exactly. The obvious judges are the frontier models this project shows disagreeing with the experts on 71 of 1,000 lines.
4. **On contested labels, "model error" is often disagreement between people.** The author relabelled the best prompted cell's disagreements blind:
   - On Claude's 47 false alarms, the author sided with the experts 35 times.
   - On Claude's 24 misses, the author sided with Claude 21 times.
   - On the hardest lines, agreement with the experts was Cohen's κ 0.16.
5. **Synthetic lines teach what the human labels leave open, and lose where the labels push the other way.**
   - "diff" appears in none of the 10,919 expert training lines. After 56 teacher-written "diff" put-downs, v2 flags `jg diff`; v1 didn't. Both held with a second training seed.
   - Sarcasm looks like praise, and the expert data labels praise-shaped lines fine: 10 of the 11 lines pairing words like "gj" or "nice" with a role are labelled not toxic. After 57 teacher-written sarcastic lines, v2 still misses all 3 sarcastic pressure lines, with both seeds.
   - At 4 times the dose (v3), the labels pushed back. v3 caught one sarcastic pressure line, and started flagging benchmark lines the experts call fine: about half of its 14 new false alarms are sarcasm or taunts, such as `ty for lose` and `soo easy for you`. Precision fell 0.118, below the bar.
   - That is one example each way plus one dose test, with one training seed for v3, so it's a well-supported hypothesis rather than a rule.

## Insights

- **Precision comes first; recall comes with data and luck.** At 1,364 rows both training seeds score precision near 0.8, but recall is 0.216 with one seed and 0.510 with the other. From 5,459 rows up, the two seeds agree within 0.03 F1.
- **Training noise is as big as the data fix.** Retraining v1 with only the seed changed moved precision by 0.056 (CI 0.01 to 0.11). The teacher lines moved it by +0.034 with one seed and −0.004 with the other. One training run per variant would have credited the teacher lines with a precision gain they don't make.
- **The small model sees sarcasm; its labels taught it not to flag it.** A probe on 13 sarcastic and 69 sincere real chat lines:
  - Asked "is this line sarcastic?", v2 is as good as Claude: to catch all 13 sarcastic lines, both also call 26 sincere lines sarcastic. The untrained 1.7B model calls 52, so size mattered before training.
  - Asked whether a line is toxic, v2 gives sarcasm a median probability of 0.095, against 0.755 for real toxic lines.
  - The experts labelled 7 of the 10 sarcastic test-split lines fine (`ty for lose`, `ty for no help`). The gap with Claude is the standard, not the model.
- **A fifth of the data from the teacher costs about 0.1 precision, twice.** v2-small (teacher lines 20% of its rows) lost 0.095 against v1 at the same expert size; v3 (19.8%) lost 0.118 against v2. Both intervals clear zero. At 5.8% (v2) precision didn't move. The teacher lines are 40% toxic, so a big dose also raises the toxic base rate the model sees.
- **Robustness doesn't come from more expert data.** Pressure-set accuracy stays between 0.625 and 0.700 at every size, in both seeds.
- **Expert labels teach the benchmark, not the written rules.**
  - The paper's fine-tuned BERT has the best `E1000` score here, F1 0.770, though choosing its epoch and threshold on the test split flatters that.
  - On the pressure set it gets 25/40, the untrained Qwen's score, and catches 2 of the 11 toxic lines. v1, trained on the same labels, gets 27/40.
  - The pressure-set gains came from the teacher lines (v2, 30/40) and from prompting (Claude, 38/40).
- **A quantisation artefact can fake a huge fine-tuning gain.** Scored with a 4-bit base, the untrained model failed to answer 933 of 1,000 lines, so any tuned model would have looked transformative. All scores here use bf16.
- **Borderline answers can depend on the batch.** In bf16 with left padding, one benchmark line got a different untrained answer in a batch of 8 than in the eval's batch of 32. The demo uses only lines whose answers held both ways.
- **Small is also cheap and fast.** The tuned model labels a line in about 0.13 s in batches on a laptop GPU. The best frontier cell takes a median 1.3 s and costs about $940 per million messages.

## Experts

- **Ave, Pekaric, Frohner and Apruzzese**, authors of [L2DTnH](https://arxiv.org/abs/2604.10175) (2026).
  - **Main claim:** prompted GPT-4o reached precision 0.35 on League chat, against 0.86 for a fine-tuned BERT.
  - **Why they matter here:** their dataset and labelling protocol are this project's ground truth.
  - **Caveat:** their GPT-4o numbers did not reproduce. Their subset couldn't be rebuilt, and our gpt-4o run scored precision 0.615 on `E1000`.
  - **Caveat:** their fine-tuned BERT chose its best epoch and its threshold on the test split, so its 0.86 precision and 0.76 F1 are optimistic. Rescored here from their published weights, it gets 0.847 and 0.770 on `E1000` and 25/40 on the pressure set.
- **Dettmers, Pagnoni, Holtzman and Zettlemoyer**, [QLoRA](https://arxiv.org/abs/2305.14314) (2023). Training LoRA adapters on a 4-bit base is what fits a 1.7B fine-tune on an 8 GB laptop GPU.
- **Hu et al.**, [LoRA](https://arxiv.org/abs/2106.09685) (2021). Low-rank adapters are why the shipped model is a 35 MB file on top of a public base.

## Knowledge tree

### 1. The data

- **Sources:** the L2DTnH paper and repository; [docs/DATASET.md](DATASET.md).
- **Facts:**
  - 12,136 training messages (1,118 toxic) and 3,035 test messages (280 toxic).
  - 8 players with 6 to 20 years of League experience labelled the first 5,000; a message was toxic if 2 flagged it. Fleiss' κ was 0.62, and 4,704 of the 5,000 were unanimous.
  - A second, undescribed review pass changed 630 labels.
- **Summary:** a large, expert-made standard, but a lenient-to-strict mix. The 2-of-8 rule leans toxic on borderline lines, and the labellers saw context the model never sees.

### 2. The prompting ceiling (Ablation 1)

- **Sources:** [results/ablation1.md](../results/ablation1.md); [docs/CEILING_REPORT.md](CEILING_REPORT.md).
- **Facts:**
  - The grid was 2 model families × 3 prompts on 1,000 benchmark lines.
  - The best cell was Claude Opus 5.5 zero-shot: precision 0.624, F1 0.687. On the full test split it scored precision 0.594 and F1 0.666.
  - No cell met the bar.
- **Summary:** current frontier models over-flag short trash talk the experts let pass, and miss abuse that only shows in context.

### 3. Base vs tuned

- **Source:** [results/slm_results.md](../results/slm_results.md).
- **Facts:**
  - Untrained Qwen3-1.7B: precision 0.466, F1 0.468, 25/40 on the pressure set.
  - v1: 0.823, 0.718, 27/40.
  - v2: 0.857, 0.698, 30/40. Retrained with seed 42: 0.875, 0.724, 29/40.
  - The paper's fine-tuned BERT: 0.847, 0.770, 25/40. Its benchmark score is flattered by test-split model selection.
  - On the benchmark, v2 fixes 77 of the untrained model's mistakes and makes 20 new ones.
- **Summary:** fine-tuning on the experts' labels buys precision, which is what the bar asked for most. It doesn't reach the F1 bar.

### 4. The data fix (v2) and a heavier dose (v3)

- **Sources:** [docs/SLM_REPORT.md](SLM_REPORT.md); `data/sft/teacher_v2_meta.json`.
- **Facts:**
  - The teacher wrote 720 lines across 12 categories. Gates removed 4 for leakage, and the critic disputed 40, leaving 676, 5.8% of v2's data.
  - On the pressure set, v2 has 3 false alarms against v1's 5, and catches 4 toxic lines against v1's 3.
  - With a second training seed, only the `jg diff` catch repeats; the two fixed false alarms don't.
  - v3, with the teacher lines 4 times: precision 0.739, recall 0.667, F1 0.701, 29/40. It catches 1 of 3 sarcastic pressure lines.
- **Summary:** the targeted data taught "diff" put-downs. It did not teach sarcasm, taunts like `gg ez`, or polite put-downs. A heavier dose taught a little sarcasm at a cost in precision the bar doesn't allow.

### 5. Why sarcasm is missed

- **Source:** [results/sarcasm_probe.md](../results/sarcasm_probe.md) (`src/run_sarcasm_probe.py`).
- **Facts:**
  - The lines: 13 sarcastic, 69 sincere and 7 unclear, from the pressure set and the test split. The test-split marks were made by Claude, the coding assistant.
  - Asked "is it sarcastic?", the models' false calls when catching all 13: untrained 52, v1 33, v2 26, v3 19, Claude 26.
  - Median toxic probability for sarcastic lines: v1 0.010, v2 0.095, v3 0.407; for real toxic benchmark lines, v2 0.755.
  - v2 at a 0.2 threshold flags 6 of the 13, at precision 0.621.
- **Summary:** the tuned model sees sarcasm and was taught it isn't toxic. Catching it costs precision against the experts, at any threshold.

### 6. Data efficiency (Ablation 2)

- **Source:** [results/slm_results.md](../results/slm_results.md#data-efficiency-curve).
- **Facts:**
  - F1 by training rows, seed 3407 / seed 42: 0.344 / 0.619 at 1,364; 0.637 / 0.678 at 2,729; 0.709 / 0.705 at 5,459; 0.718 / 0.690 at 10,919.
  - In both seeds, 5,459 rows score within noise of 10,919.
- **Summary:** steep and seed-dependent up to about 5,500 rows (496 toxic), flat after. That is the minimum N this project justifies. It is where more of the same data stops helping, not where the bar is met. No size between 2,729 and 5,459 was tried.

## Open questions

- How does v2 score on the staff held-out set?
- Are v3's new false alarms errors? Relabelled under the written rules, the sarcasm-shaped lines the experts call fine might count as v3's catches.
- Would teacher lines the experts' labels agree with teach sarcasm without the precision cost?
- Do the gains hold on lines written by someone other than the teacher model?
- The question the project started from: can a model trained mostly on teacher-written chat match one trained on real labelled chat?
