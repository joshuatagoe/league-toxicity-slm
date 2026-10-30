# League toxicity SLM

A small language model that labels League of Legends chat messages as toxic or non-toxic, built for Gauntlet's Week 7 assignment ("Train your own small model").

The first milestone is the **prompt-ceiling gate**: do current frontier models still misjudge League chat, as GPT-4o did in the L2DTnH paper? The approved plan is in [docs/PLAN.md](docs/PLAN.md).

**Gate result (2026-10-29): passed.** The best of six model × prompt cells (Claude Opus 5.5, zero-shot) reached precision 0.594 and F1 0.666 on the full test split, below the bar. Details in [docs/CEILING_REPORT.md](docs/CEILING_REPORT.md).

## Behaviour spec

> Given one League of Legends chat line, the model replies with exactly one word, `toxic` or `nontoxic`. It says `toxic` only when the line attacks, mocks or demeans a player (insults, including disguised or non-English ones; sarcasm or taunts aimed at someone; "diff" put-downs), and `nontoxic` for game talk, callouts, self-criticism, quoting or reporting someone, swearing at the game, friendly banter, and telling a flamer to stop.

**How a reply is marked.** A reply passes if it is exactly one of the two words and matches the line's label; anything else (another word, an explanation, no reply) fails. The full rules, with examples, are in [docs/LABELING_RULES.md](docs/LABELING_RULES.md).

| Measure | Eval set | Labels from |
| --- | --- | --- |
| Spec adherence: precision and F1 for `toxic` | `E1000`, the first 1,000 test messages | The L2DTnH expert annotators |
| Robustness: accuracy | `PRESSURE`, 40 hand-picked hard lines, 4 per rule | The project author, under the written rules |

The expert labels and the written rules disagree on some lines (see [results/relabel_analysis.md](results/relabel_analysis.md)). `E1000` is scored against the experts so the numbers stay comparable with the paper; the pressure set checks the rules the experts didn't write down.

## Model checkpoint and pinned versions

| What | Pinned to |
| --- | --- |
| Tuned model (v2 LoRA adapter, public) | [Juchen/league-toxicity-qwen3-1.7b](https://huggingface.co/Juchen/league-toxicity-qwen3-1.7b) at commit `80e3d1bac9505024f49cf3439926f2be7d2e809d` |
| Base model | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) at commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Eval code | this repo at commit `3745c5e9894ea5b6d5d57b6e945fd990ffd29154` |

One command per eval set, in the training environment (`requirements-train.txt`):

```
python eval.py --model Juchen/league-toxicity-qwen3-1.7b --revision 80e3d1bac9505024f49cf3439926f2be7d2e809d --eval-set data/eval/E1000.jsonl
```

Use `--eval-set data/pressure/PRESSURE.jsonl` for the pressure set, or `--model Qwen/Qwen3-1.7B` for the untrained base. Run from the Hub this way, the adapter gave the same answer as the local run on all 1,000 `E1000` messages and all 40 pressure lines (`runs/qwen3-1.7b-v2-hub__*`).

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
