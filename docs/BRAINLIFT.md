# BrainLift: does data beat prompting for League chat toxicity?

**Draft for the early submission, 2026-10-29.** The final version comes with the final submission, after a second seed per training size and a score on the staff held-out set. Every number here comes from [results/slm_results.md](../results/slm_results.md), [results/ablation1.md](../results/ablation1.md) or [results/relabel_analysis.md](../results/relabel_analysis.md).

## Purpose

Work out when a narrow judgement is better taught with labelled data than asked for in a prompt. The test case is one League of Legends chat line, labelled `toxic` or `nontoxic`, where a false flag punishes an innocent player, so precision matters most.

- **In scope:** single-line classification; precision-first moderation; what a small model learns from expert labels versus teacher-written lines; how much data it takes.
- **Out of scope:** moderation with the surrounding chat or game state; deciding punishments; languages other than English.

## Thesis

A community's standard lives in its labels, not in a description of it. Qwen3-1.7B, trained on about 11,000 expert-labelled League chat lines, holds that standard more precisely than the best frontier model prompted to apply it. Teacher-written lines can add to that standard, but only where the human labels don't push the other way. At 6% of the data and one epoch, they taught "diff" put-downs, which the expert data never mentions. They did not teach sarcasm, which looks like the praise the experts label fine.

## Did data → behaviour hold?

| Behaviour | Data change | Result | Held? |
| --- | --- | --- | --- |
| Flag real toxic chat precisely | 10,919 expert-labelled rows (v1) | Precision 0.466 → 0.823, a gain of +0.36 (95% CI +0.26 to +0.46). F1 0.468 → 0.718 | **Yes** |
| Meet the bar: precision ≥ 0.80 and F1 ≥ 0.75 | Same | v1 reaches precision 0.823, F1 0.718; the shipped v2 reaches 0.857 and 0.698 | **Half:** the precision part holds, the F1 part doesn't |
| Catch hostility without insult words | Plus 676 teacher lines aimed at it (v2) | 3 → 4 of 11 toxic pressure lines caught; none of the 3 sarcastic ones | **No** |
| Leave harmless lines with hostile-looking words alone | Same teacher lines | False alarms 5 → 3 of 29 pressure lines. Pressure set +3 lines overall (CI 0.00 to +0.15); benchmark F1 −0.021 (CI −0.07 to +0.03) | **Weakly**, within noise |
| Improve with more of the same data | 5,459 → 10,919 expert rows | F1 +0.010 (CI −0.03 to +0.05); pressure score flat | **No gain** past about 5,500 rows |

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
   - "diff" appears in none of the 10,919 expert training lines. After 56 teacher-written "diff" put-downs, v2 flags `jg diff`; v1 didn't.
   - Sarcasm looks like praise, and the expert data labels praise-shaped lines fine: 10 of the 11 lines pairing words like "gj" or "nice" with a role are labelled not toxic. After 57 teacher-written sarcastic lines, v2 still misses all 3 sarcastic pressure lines.
   - This is one example each way, at one dose, so it's a hypothesis rather than a rule. The final version tests a heavier dose.

## Insights

- **The model learns to stay quiet before it learns what to flag.** At 1,364 rows, precision is already 0.846 but recall is 0.216. Recall climbs with data (0.500, 0.608, 0.637); precision stays above 0.82.
- **Robustness doesn't come from more expert data.** Pressure-set accuracy stays between 0.675 and 0.700 at every size.
- **A quantisation artefact can fake a huge fine-tuning gain.** Scored with a 4-bit base, the untrained model failed to answer 933 of 1,000 lines, so any tuned model would have looked transformative. All scores here use bf16.
- **Borderline answers can depend on the batch.** In bf16 with left padding, one benchmark line got a different untrained answer in a batch of 8 than in the eval's batch of 32. The demo uses only lines whose answers held both ways.
- **Small is also cheap and fast.** The tuned model labels a line in about 0.13 s in batches on a laptop GPU. The best frontier cell takes a median 1.3 s and costs about $940 per million messages.

## Experts

- **Ave, Pekaric, Frohner and Apruzzese**, authors of [L2DTnH](https://arxiv.org/abs/2604.10175) (2026).
  - **Main claim:** prompted GPT-4o reached precision 0.35 on League chat, against 0.86 for a fine-tuned BERT.
  - **Why they matter here:** their dataset and labelling protocol are this project's ground truth.
  - **Caveat:** their GPT-4o numbers did not reproduce. Their subset couldn't be rebuilt, and our gpt-4o run scored precision 0.615 on `E1000`.
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
  - v2: 0.857, 0.698, 30/40.
  - On the benchmark, v2 fixes 77 of the untrained model's mistakes and makes 20 new ones.
- **Summary:** fine-tuning on the experts' labels buys precision, which is what the bar asked for most. It doesn't reach the F1 bar.

### 4. The data fix (v2)

- **Sources:** [docs/SLM_REPORT.md](SLM_REPORT.md); `data/sft/teacher_v2_meta.json`.
- **Facts:**
  - The teacher wrote 720 lines across 12 categories. Gates removed 4 for leakage, and the critic disputed 40, leaving 676, 5.8% of v2's data.
  - On the pressure set, v2 has 3 false alarms against v1's 5, and catches 4 toxic lines against v1's 3.
- **Summary:** the targeted data fixed two false alarms and taught "diff" put-downs. It did not teach sarcasm, taunts like `gg ez`, or polite put-downs.

### 5. Data efficiency (Ablation 2)

- **Source:** [results/slm_results.md](../results/slm_results.md#data-efficiency-curve).
- **Facts:**
  - F1 by training rows: 0.344 at 1,364; 0.637 at 2,729; 0.709 at 5,459; 0.718 at 10,919.
  - One seed per size.
- **Summary:** steep up to about 5,500 rows, flat after. That's an estimate from four sizes, not a measured minimum.

## Open questions for the final submission

- Does a second seed per size keep the curve's shape?
- How does v2 score on the staff held-out set?
- Does weighting the teacher lines more, or training on them for more epochs, teach sarcasm without costing precision?
- Do the gains hold on lines written by someone other than the teacher model?
- The question the project started from: can a model trained mostly on teacher-written chat match one trained on real labelled chat?
