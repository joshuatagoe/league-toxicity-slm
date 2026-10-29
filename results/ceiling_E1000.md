# Prompt-ceiling results: E1000

Bar: precision ≥ 0.80 and F1 ≥ 0.75 (toxic = positive class). Failed replies (refused, unparseable, truncated, errored) count as wrong answers. Intervals are 95% bootstrap, 1,000 resamples.

| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | zero_shot_p1 | low | 0.624 (0.54–0.71) | 0.765 (0.69–0.85) | 0.687 (0.62–0.76) | no | 0 | 1295 ms | $943 |
| gpt-4o | zero_shot_p1 | low | 0.615 (0.52–0.72) | 0.627 (0.53–0.73) | 0.621 (0.54–0.70) | no | parse_fail 1 | 841 ms | $226 |
| gpt-5.6-sol | few_shot | low | 0.419 (0.35–0.49) | 0.882 (0.82–0.94) | 0.568 (0.50–0.63) | no | 0 | 1076 ms | $1,950 |
| gpt-5.6-sol | zero_shot_p1 | low | 0.417 (0.35–0.49) | 0.863 (0.79–0.93) | 0.562 (0.50–0.63) | no | 0 | 1093 ms | $657 |
| baseline:keyword | zero_shot_p1 | na | 0.547 (0.45–0.64) | 0.569 (0.48–0.67) | 0.558 (0.48–0.64) | no | 0 | 0 ms | n/a |
| gpt-5.6-sol | structured | low | 0.415 (0.35–0.48) | 0.833 (0.76–0.90) | 0.554 (0.48–0.61) | no | 0 | 1472 ms | $2,401 |
| claude-opus-5-5 | few_shot | low | 0.351 (0.30–0.41) | 0.892 (0.83–0.95) | 0.504 (0.44–0.57) | no | 0 | 1384 ms | $2,824 |
| claude-opus-5-5 | structured | low | 0.346 (0.29–0.41) | 0.873 (0.80–0.93) | 0.496 (0.43–0.56) | no | 0 | 1725 ms | $3,383 |
| baseline:always-nontoxic | zero_shot_p1 | na | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | 0.000 (0.00–0.00) | no | 0 | 0 ms | n/a |

Paper rows for reference (measured on a different 707-message subset, so the comparison is approximate):

| Source | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Paper: GPT-4o, P1 (toxic) | 0.349 | 0.923 | 0.507 |
| Paper: GPT-4o, P2 (inappropriate) | 0.366 | 0.667 | 0.473 |
