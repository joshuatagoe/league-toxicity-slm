# Prompt-ceiling results: FULL

Bar: precision ≥ 0.80 and F1 ≥ 0.75 (toxic = positive class). Failed replies (refused, unparseable, truncated, errored) count as wrong answers. Intervals are 95% bootstrap, 1,000 resamples.

| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | zero_shot_p1 | low | 0.594 (0.54–0.64) | 0.757 (0.71–0.81) | 0.666 (0.62–0.70) | no | 0 | 1301 ms | $939 |
| baseline:keyword | zero_shot_p1 | na | 0.556 (0.51–0.61) | 0.568 (0.51–0.62) | 0.562 (0.51–0.60) | no | 0 | 0 ms | n/a |
| baseline:always-nontoxic | zero_shot_p1 | na | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | no | 0 | 0 ms | n/a |

Paper rows for reference (measured on a different 707-message subset, so the comparison is approximate):

| Source | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Paper: GPT-4o, P1 (toxic) | 0.349 | 0.923 | 0.507 |
| Paper: GPT-4o, P2 (inappropriate) | 0.366 | 0.667 | 0.473 |
