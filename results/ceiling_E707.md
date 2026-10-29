# Prompt-ceiling results: E707

Bar: precision ≥ 0.80 and F1 ≥ 0.75 (toxic = positive class). Failed replies (refused, unparseable, truncated, errored) count as wrong answers. Intervals are 95% bootstrap, 1,000 resamples.

| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:keyword | zero_shot_p1 | na | 0.526 (0.42–0.64) | 0.625 (0.51–0.74) | 0.571 (0.47–0.67) | no | 0 | 0 ms | n/a |
| baseline:always-nontoxic | zero_shot_p1 | na | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | no | 0 | 0 ms | n/a |

Paper rows for reference (a different 707-message subset, so the comparison is approximate):

| Source | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Paper: GPT-4o, P1 (toxic) | 0.349 | 0.923 | 0.507 |
| Paper: GPT-4o, P2 (inappropriate) | 0.366 | 0.667 | 0.473 |
