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

## Caveats

- **Part of the gap is the labels.** By our reading, many of the 47 false alarms are plainly hostile (`so useless eve`, `that btch`, `just l2p`), yet are labelled non-toxic. The planned relabelling of the 71 disagreements will measure how much of the gap is annotation rather than model error.
- **The paper's GPT-4o numbers weren't reproduced.** Our gpt-4o run with the paper's exact prompt scored precision 0.615 and recall 0.627 on E1000, against the paper's 0.349 and 0.923 on its own 707-message subset. The subset differs and the `gpt-4o` alias may point to a different snapshot, so this project's numbers stand on their own rather than on the paper's.
- **Robustness column still to come.** The brief's table also wants Robustness per cell. That needs the hand-written pressure set (sarcasm, quotes, self-criticism, harmless trigger words), run through the same six cells.
- **Run naming.** The gpt-4o run is named with effort `low`, but no effort is sent to gpt-4o; it runs at temperature 0, as in the paper.
