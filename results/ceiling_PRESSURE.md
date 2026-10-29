# Prompt-ceiling results: PRESSURE

Bar: precision ≥ 0.80 and F1 ≥ 0.75 (toxic = positive class). Failed replies (refused, unparseable, truncated, errored) count as wrong answers. Intervals are 95% bootstrap, 1,000 resamples.

| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | zero_shot_p1 | low | 0.846 (0.62–1.00) | 1.000 (1.00–1.00) | 0.917 (0.77–1.00) | yes | 0 | 1383 ms | $1,153 |
| gpt-5.6-sol | few_shot | low | 0.647 (0.40–0.88) | 1.000 (1.00–1.00) | 0.786 (0.57–0.94) | no | 0 | 1248 ms | $1,948 |
| gpt-5.6-sol | structured | low | 0.647 (0.40–0.88) | 1.000 (1.00–1.00) | 0.786 (0.57–0.93) | no | 0 | 1642 ms | $2,421 |
| claude-opus-5-5 | few_shot | low | 0.579 (0.36–0.80) | 1.000 (1.00–1.00) | 0.733 (0.53–0.89) | no | 0 | 1842 ms | $2,848 |
| claude-opus-5-5 | structured | low | 0.579 (0.37–0.80) | 1.000 (1.00–1.00) | 0.733 (0.54–0.89) | no | 0 | 1893 ms | $3,223 |
| gpt-5.6-sol | zero_shot_p1 | low | 0.556 (0.31–0.75) | 0.909 (0.67–1.00) | 0.690 (0.44–0.84) | no | 0 | 1153 ms | $600 |
| baseline:keyword | zero_shot_p1 | na | 0.111 (0.00–0.33) | 0.091 (0.00–0.31) | 0.100 (0.00–0.29) | no | 0 | 0 ms | n/a |
| baseline:always-nontoxic | zero_shot_p1 | na | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | no | 0 | 0 ms | n/a |

Paper rows for reference (measured on a different 707-message subset, so the comparison is approximate):

| Source | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Paper: GPT-4o, P1 (toxic) | 0.349 | 0.923 | 0.507 |
| Paper: GPT-4o, P2 (inappropriate) | 0.366 | 0.667 | 0.473 |
