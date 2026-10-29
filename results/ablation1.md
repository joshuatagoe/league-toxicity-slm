# Ablation 1: prompt ceiling

Spec adherence is F1 for the toxic class on E1000 (1,000 test messages, the experts' labels), with its 95% bootstrap interval. Robustness is accuracy on the 40-message pressure set (labelled by the project author; 10 toxic, 30 not). Bar: precision ≥ 0.80 and F1 ≥ 0.75 on E1000.

| Model | Prompt | Spec adherence (F1, E1000) | Precision | Recall | Robustness (accuracy, pressure set) | Pressure F1 |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5-5 | zero_shot_p1 | 0.687 (0.62–0.76) | 0.624 | 0.765 | 0.950 | 0.917 |
| gpt-5.6-sol | few_shot | 0.568 (0.50–0.63) | 0.419 | 0.882 | 0.850 | 0.786 |
| gpt-5.6-sol | zero_shot_p1 | 0.562 (0.50–0.63) | 0.417 | 0.863 | 0.775 | 0.690 |
| baseline:keyword | zero_shot_p1 | 0.558 (0.48–0.64) | 0.547 | 0.569 | 0.550 | 0.100 |
| gpt-5.6-sol | structured | 0.554 (0.48–0.61) | 0.415 | 0.833 | 0.850 | 0.786 |
| claude-opus-5-5 | few_shot | 0.504 (0.44–0.57) | 0.351 | 0.892 | 0.800 | 0.733 |
| claude-opus-5-5 | structured | 0.496 (0.43–0.56) | 0.346 | 0.873 | 0.800 | 0.733 |
| baseline:always-nontoxic | zero_shot_p1 | 0.000 (0.00–0.00) | 0.000 | 0.000 | 0.725 | 0.000 |
