# Prompt-ceiling results: PRESSURE

Bar: precision ≥ 0.80 and F1 ≥ 0.75 (toxic = positive class). Failed replies (refused, unparseable, truncated, errored) count as wrong answers. Intervals are 95% bootstrap, 1,000 resamples.

| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | zero_shot_p1 | low | 0.769 (0.53–1.00) | 1.000 (1.00–1.00) | 0.870 (0.69–1.00) | no | 0 | 1383 ms | $1,153 |
| gpt-5.6-sol | few_shot | low | 0.588 (0.35–0.82) | 1.000 (1.00–1.00) | 0.741 (0.52–0.90) | no | 0 | 1248 ms | $1,948 |
| gpt-5.6-sol | structured | low | 0.588 (0.35–0.82) | 1.000 (1.00–1.00) | 0.741 (0.52–0.90) | no | 0 | 1642 ms | $2,421 |
| claude-opus-5-5 | few_shot | low | 0.526 (0.31–0.75) | 1.000 (1.00–1.00) | 0.690 (0.47–0.86) | no | 0 | 1842 ms | $2,848 |
| claude-opus-5-5 | structured | low | 0.526 (0.32–0.76) | 1.000 (1.00–1.00) | 0.690 (0.48–0.86) | no | 0 | 1893 ms | $3,223 |
| gpt-5.6-sol | zero_shot_p1 | low | 0.500 (0.26–0.72) | 0.900 (0.67–1.00) | 0.643 (0.38–0.82) | no | 0 | 1153 ms | $600 |
| baseline:keyword | zero_shot_p1 | na | 0.111 (0.00–0.33) | 0.100 (0.00–0.33) | 0.105 (0.00–0.31) | no | 0 | 0 ms | n/a |
| baseline:always-nontoxic | zero_shot_p1 | na | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | no | 0 | 0 ms | n/a |

Paper rows for reference (measured on a different 707-message subset, so the comparison is approximate):

| Source | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Paper: GPT-4o, P1 (toxic) | 0.349 | 0.923 | 0.507 |
| Paper: GPT-4o, P2 (inappropriate) | 0.366 | 0.667 | 0.473 |
