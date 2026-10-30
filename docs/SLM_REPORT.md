# Small model report: base vs tuned, the data fix and the data-efficiency curve

Run 2026-10-29. Every number below comes from [results/slm_results.md](../results/slm_results.md), which `scripts/slm_results.py` builds from the per-message run files in `runs/` and the training logs in [results/train_logs/](../results/train_logs/). Models are Qwen3-1.7B with a LoRA adapter trained by QLoRA; all are scored in bf16.

## Summary

- **Fine-tuning works on the benchmark.** Trained on the 10,919 expert-labelled training messages, the 1.7B model goes from F1 0.468 to 0.718 on `E1000` and from precision 0.466 to 0.823. Both gains have intervals well clear of zero.
- **The tuned model clears the precision bar and misses the F1 bar.** The shipped model (v2) scores precision 0.857 and F1 0.698 against a bar of 0.80 and 0.75. Claude Opus 5.5's best prompt scored precision 0.624 and F1 0.687, so the 1.7B model matches the best frontier cell on F1 and raises precision by 0.23. It runs at about 0.13 s per message in batches of 32 on a laptop GPU, against the frontier cell's median of 1.3 s per message.
- **The data fix helped a little, not enough.** 676 teacher-written lines aimed at v1's failures moved the pressure set from 27 to 30 of 40 and left the benchmark where it was. Sarcasm and taunts are still mostly missed.
- **More expert data stopped helping at about 5,500 rows in this curve.** Doubling from 5,459 to 10,919 rows changed F1 by +0.010 (interval −0.03 to +0.05). No size reached the F1 bar.

## Base vs tuned

| Model | Precision | Recall | F1 (95% CI) | Pressure set |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B, untrained | 0.466 | 0.471 | 0.468 (0.38–0.55) | 25/40 |
| v1, 10,919 expert rows | 0.823 | 0.637 | 0.718 (0.63–0.78) | 27/40 |
| **v2, plus 676 teacher rows (shipped)** | **0.857** | 0.588 | 0.698 (0.61–0.77) | **30/40** |
| Claude Opus 5.5, best prompt | 0.624 | 0.765 | 0.687 (0.62–0.76) | 38/40 |

Paired against the untrained model on the same messages, v2 gains +0.229 F1 (+0.13 to +0.33) and +0.391 precision (+0.28 to +0.50). On the pressure set it gains 5 lines (+0.03 to +0.23 accuracy). The untrained model's 25/40 comes mostly from saying `nontoxic` to harmless lines: it catches 1 of the 11 toxic pressure lines.

Training changed what the model is for. Untrained, it flags about as many lines as it should but picks the wrong ones (precision and recall both near 0.47). Tuned, it flags fewer lines and is right about 6 times in 7, which is what the precision bar asks for: a false flag punishes an innocent player.

## The failure, and the data fix (v2)

**What v1 got wrong.** On the pressure set v1 caught 3 of the 11 toxic lines and raised 5 false alarms on 29 harmless ones. The two patterns:

- **It misses hostility without insult words:** sarcasm at a teammate (`wow great gank jungle, really helpful`), taunts (`gg ez`) and polite or disguised put-downs.
- **It flags harmless lines that contain hostile-looking words:** quoting or reporting someone else's insult, and complaints about the game.

A likely reason: the expert-labelled data has few lines of the first kind, so v1 keyed on insult words.

**The fix was data only.** Claude Opus 5.5 wrote 720 lines across 12 categories aimed at those two patterns: 5 toxic, 7 harmless. Each category's label comes from the written rules ([docs/LABELING_RULES.md](LABELING_RULES.md)). The lines went through four gates:

- deduplication;
- a length check;
- a leakage check against every eval and pressure message (4 dropped);
- a GPT-5.6 Sol critic applying the same written rules (40 disputed lines left out).

The remaining 676 lines (269 toxic) were mixed into the v1 data. Prompt, format and training settings stayed the same. Total cost: $0.81.

**Result: small gains, mostly within noise.**

| Paired comparison | F1 on E1000 | Precision | Pressure set |
| --- | --- | --- | --- |
| v2 vs v1 | −0.021 (−0.07 to +0.03) | +0.034 (−0.03 to +0.10) | +3 lines (+0.00 to +0.15) |
| v2-small vs v1, both on 2,729 expert rows | +0.048 (−0.01 to +0.11) | −0.095 (−0.18 to −0.02) | −1 line (−0.12 to +0.05) |

- On the pressure set v2 cut false alarms from 5 to 3. It no longer flags `fucking lag again` (swearing at the game) or `cry more top` (a line the rules leave fine because it depends on unseen context). It caught one more toxic line, `jg diff`, a kind of put-down the teacher data targeted.
- **The miss is not fixed.** v2 still catches only 4 of 11 toxic pressure lines:
  - none of the 3 sarcastic lines, including `wow great gank jungle, really helpful`;
  - 1 of 3 taunts (it answers `nontoxic` to `gg ez`);
  - neither polite put-down (`go play some other game pls`, `have u considered tft`).
- The benchmark didn't move, which is expected: `E1000` holds few sarcastic or taunting lines.
- **v2-small** puts the same 676 lines next to only 2,729 expert rows, so a fifth of its data is teacher-written. Its precision dropped by 0.095, below the bar (0.785), with no gain on the pressure set. A bigger share of teacher data made the model flag more, but not more accurately.

**Why the fix fell short.**
- 676 lines are 6% of v2's training data, seen once (1 epoch).
- The 10,919 expert rows outnumber them 16 to 1.
- The teacher lines came from one model's idea of sarcasm. The pressure set's lines were written separately, so the two sets share a style only loosely, by design.

A second fix would weight the teacher lines more heavily or train for more epochs on them, and score the result on lines from a different writer.

**What was shipped.** v2 has the highest precision and the best pressure score of the tuned models, so it is the public checkpoint ([README](../README.md#model-checkpoint-and-pinned-versions)).

## Data-efficiency curve (Ablation 2)

![Data-efficiency curve](../results/data_efficiency.svg)

| Training rows | Spec adherence (F1, E1000) | Precision | Robustness (pressure accuracy) |
| --- | --- | --- | --- |
| 1,364 | 0.344 | 0.846 | 0.700 |
| 2,729 | 0.637 | 0.879 | 0.675 |
| 5,459 | 0.709 | 0.849 | 0.700 |
| 10,919 | 0.718 | 0.823 | 0.675 |

**Setup.** Four v1 checkpoints were trained on nested prefixes of the shuffled expert-labelled file (N/8, N/4, N/2, N; each contains the smaller ones), with identical settings and one seed.

**Threshold rule.** The bar was fixed in the README before any run: precision ≥ 0.80 and F1 ≥ 0.75 on `E1000`. The rule for choosing N, "the smallest N that meets the bar", is the guide's default. It was not written into the repo before the curve was run.

**Reading the curve:**

- **No size meets the bar,** so under that rule there is no minimum viable N on this data. Every size from 1,364 rows up clears the precision bar. None reaches F1 0.75.
- **F1 climbs steeply, then flattens.** It rises by 0.29 from 1,364 to 2,729 rows and by 0.07 from 2,729 to 5,459. From 5,459 to 10,919 it rises by 0.010, with an interval from −0.03 to +0.05. Half the data gives the same F1 within noise.
- **Robustness doesn't improve with size.** It stays between 0.675 and 0.700, one line either way. A likely reason is that the pressure behaviours are rare in the expert data.

**About 5,500 expert-labelled rows is where F1 stops improving in this curve.** With four sizes and one seed, that is an estimate, not a measured minimum: no size between 2,729 and 5,459 was tried. Different data may help more than more of the same, but the one test of that so far (v2) didn't move F1. For the final submission, the curve gets a second seed per size and a score on the staff held-out set.

## Scoring: exact match, not an LLM judge

The brief scores outputs with an LLM-as-judge. Here every eval line already has a human label, and the model must answer with one word. So a reply is compared with the label directly: it passes only if it is exactly `toxic` or `nontoxic` and matches.

A judge would add a second model's mistakes and cost to a check that a string comparison does exactly. The obvious judges are the frontier models, and Ablation 1 shows them disagreeing with the experts on 71 of 1,000 lines.

The brief's "raw transcripts" are the per-message run files in `runs/*.jsonl`. Each line holds the message, the label, the parsed prediction and the model's raw reply, so any score can be recomputed or audited line by line.

## Smoke test: the whole loop on a small batch

`bash train/smoke.sh` runs every stage on a small batch in about 4 minutes, and its output is saved in [results/smoke.log](../results/smoke.log):

1. The teacher writes 20 lines in two categories, through the same gates. The critic is GPT-5.6 Sol. Cost: $0.03.
2. The 20 lines are mixed with 240 expert rows.
3. QLoRA trains Qwen3-0.6B for 16 steps.
4. Base and tuned models are scored on 50 benchmark lines and the pressure set.

The run of 2026-10-29 completed every stage with no errors. The log also prints the trained tokens: only `nontoxic<|im_end|>`, which confirms the loss falls on the answer alone.

The smoke model is not meant to be good. After 16 steps on 11% toxic data, it answers `nontoxic` to everything. It shows the pipeline runs, not what the model learns.

## Caveats

- **One seed per size.** The intervals cover sampling of eval messages, not training noise between runs.
- **The pressure set is small.** It has 40 lines, 11 of them toxic, labelled by the author. A change of 1–3 lines is within noise.
- **The two label sources disagree on some lines** ([results/relabel_analysis.md](../results/relabel_analysis.md)). v1 learns the experts' standard; the pressure set tests the written rules.
- **The first tuned scores used a 4-bit base and broke the untrained model.** It failed to answer 933 of 1,000 times. All tables here use bf16. The 4-bit runs are kept as `runs/*-nf4*` ([docs/PLAN.md](PLAN.md)).
- **The teacher is the same model family as the best frontier cell.** Its idea of sarcasm may leak a frontier model's standard into v2.
