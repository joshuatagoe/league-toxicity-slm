# Prompt-ceiling ablation (Ablation 1)

Run 2026-10-29. Generated tables: [results/ceiling_E1000.md](../results/ceiling_E1000.md) (all cells) and [results/ceiling_FULL.md](../results/ceiling_FULL.md) (best cell on all 3,035 test messages). Every number below comes from those files and the run files in `runs/`.

## Result: prompting plateaus below the bar, so the project goes ahead

The bar was precision ≥ 0.80 and F1 ≥ 0.75, set before any run. No cell reached it. The best of the six, Claude Opus 5.5 with the paper's zero-shot prompt, scored:

| Eval set | Precision | Recall | F1 |
| --- | --- | --- | --- |
| E1000 (102 toxic) | 0.624 (0.54–0.71) | 0.765 (0.69–0.85) | 0.687 (0.62–0.76) |
| FULL (280 toxic) | 0.594 (0.54–0.64) | 0.757 (0.71–0.81) | 0.666 (0.62–0.70) |

The top of the precision interval is 0.64 on the full test split, well under 0.80.

## What the grid showed

- **The newest OpenAI model is no better than a keyword list.** GPT-5.6 Sol scored F1 0.554–0.568 across the three prompts; the keyword rule scored 0.558 on E1000.
- **More instructions made both families worse.** The few-shot and structured prompts carry the annotators' guideline, including "a message counts as toxic if even a couple of experienced players would find it offensive". They raised recall and cut precision: Claude Opus 5.5 fell from 0.624 to 0.351 (few-shot) and 0.346 (structured). The simplest prompt was the best for both models.
- **Cost and speed.** The best frontier cell costs about $940 per million messages at a median 1.3 s per message.

## The failure that survives the best prompt

The best cell made 47 false alarms and 24 misses on E1000 ([errors file](../results/errors_claude-opus-5-5__zero_shot_p1__low__E1000.md)).

- **False alarms:** it flags short trash talk that the expert panel left unflagged, such as `easy`, `bg`, `gg mid`, `0/3 nasus` and `just l2p`.
- **Misses:** messages whose toxicity depends on context the message alone doesn't show, such as `GAREN TANK`, `19 assist` and `look at gangplank`. The annotators could see surrounding messages; the model sees one line.

In one sentence: a prompted frontier model holds a stricter and more context-free standard than League's expert players, so it over-flags banter and misses context-dependent abuse.

## Robustness (the brief's second column)

The full Ablation 1 table, with Spec adherence (F1 on E1000) and Robustness (accuracy on the 40-message pressure set) per model × prompt, is in [results/ablation1.md](../results/ablation1.md). The pressure set was written to trip models (sarcasm, quotes, self-criticism, harmless trigger words, banter, taunts, disguised insults) and labelled by the project author, giving 11 toxic and 29 not. The best E1000 cell is also the most robust: 38 of 40 correct. Its 2 misses (`nice ward supp, very useful`, `cry more top`) are lines whose toxicity depends on context the set doesn't give, which the author's rules label not toxic by default. So the pressure set mostly shows that frontier models handle clear-cut trick lines; their problem is the expert standard on real chat.

## Relabel check: whose standard is the gap?

The author relabelled the best cell's 71 disagreements plus 29 agreement controls, blind and with the surrounding chat ([results/relabel_analysis.md](../results/relabel_analysis.md)).

- **On Claude's 47 false alarms, the author sided with the experts 35 times** (10 with Claude, 2 unsure). These lines were labelled fine in the first pass and never revisited; under the paper's rule (toxic if at least 2 of 8 annotators flag it), "fine" means at most 1 of 8 flagged it. So the over-flagging is real: experienced players, the author included, don't treat lines like `just l2p` or `report bot and xin` as toxic.
- **On Claude's 24 misses, the author sided with Claude 21 times.** 21 of the 24 sit in the dataset's second review pass (`wrongly classified`), and the 2-of-8 rule makes a borderline line toxic. So most "missed toxicity" is lines a single experienced player wouldn't flag.
- **The author is the most lenient of the three:** of 15 controls where the experts and Claude both said toxic, the author agreed on 9. Agreement on these hard cases is low for everyone (Cohen's kappa 0.16 with the experts, 0.21 with Claude).

This changes the reading of the gap: Claude's false alarms reflect a stricter standard than League players hold, while its misses mostly reflect an inclusive labelling rule. A spec written as explicit rules (feed callouts, plain report calls, self-defence, sarcasm at a teammate) would be more consistent than either.

## Caveats

- **The labels went through two passes.** The `old_label` and `wrongly classified` columns show about 1,526 messages were re-reviewed: 365 flipped toxic to fine, 265 fine to toxic. The paper doesn't describe this pass; this reading of the columns is ours.
- **The paper's GPT-4o numbers weren't reproduced.** Our gpt-4o run with the paper's exact prompt scored precision 0.615 and recall 0.627 on E1000, against the paper's 0.349 and 0.923 on its own 707-message subset. The subset differs and the `gpt-4o` alias may point to a different snapshot, so this project's numbers stand on their own rather than on the paper's.
- **Robustness column still to come.** The brief's table also wants Robustness per cell. That needs the hand-written pressure set (sarcasm, quotes, self-criticism, harmless trigger words), run through the same six cells.
- **Run naming.** The gpt-4o run is named with effort `low`, but no effort is sent to gpt-4o; it runs at temperature 0, as in the paper.
