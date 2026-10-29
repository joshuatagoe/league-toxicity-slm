# League toxicity SLM

A small language model that labels League of Legends chat messages as toxic or non-toxic, built for Gauntlet's Week 7 assignment ("Train your own small model").

The first milestone is the **prompt-ceiling gate**: do current frontier models still misjudge League chat, as GPT-4o did in the L2DTnH paper? The approved plan is in [docs/PLAN.md](docs/PLAN.md).

**Gate result (2026-10-29): passed.** The best of six model × prompt cells (Claude Opus 5.5, zero-shot) reached precision 0.594 and F1 0.666 on the full test split, below the bar. Details in [docs/CEILING_REPORT.md](docs/CEILING_REPORT.md).

## The bar (set before any run)

| Metric (toxic = positive class) | Bar | Source |
| --- | --- | --- |
| Precision | ≥ 0.80 | A false flag punishes an innocent player; the paper's fine-tuned BERT reached 0.86 |
| F1 | ≥ 0.75 | About what the paper's fine-tuned BERT reached (0.76) |

**Decision rule.** If the best frontier model × prompt cell scores below the bar on precision or F1, prompting has a ceiling and the project goes ahead. If any cell meets both, prompting is enough and the target changes before anything else is built.

## Data

[L2DTnH](https://arxiv.org/abs/2604.10175) (Ave, Pekaric, Frohner, Apruzzese, 2026): about 15.2k English League chat messages labelled toxic or non-toxic by 8 expert players. Released under the MIT licence at [irdin-pekaric/esorics26_toxicity](https://github.com/irdin-pekaric/esorics26_toxicity), pinned here to commit `66929d13e455d529f99feee2819374a60ecd6ad9`.

The raw CSVs are not committed. `scripts/get_data.py` downloads them at the pinned commit and checks their hashes and row counts.

The messages come from real players. Use them for research only, and never to label a real person as toxic.

## Paper baseline (GPT-4o, June 2025, first 707 test messages)

| Prompt | Precision | Recall | F1 |
| --- | --- | --- | --- |
| P1 ("toxic") | 0.3495 | 0.9231 | 0.5070 |
| P2 ("inappropriate") | 0.3662 | 0.6667 | 0.4727 |
| Fine-tuned BERT (full test split) | 0.8571 | 0.6857 | 0.7619 |

Source: Table of results in [arXiv:2604.10175](https://arxiv.org/abs/2604.10175).
