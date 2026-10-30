---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- lora
- qlora
- toxicity
- content-moderation
- league-of-legends
---

# League chat toxicity: Qwen3-1.7B LoRA (v2)

A LoRA adapter for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (base commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`) that labels one League of Legends chat line `toxic` or `nontoxic`. Built for Gauntlet's Week 7 assignment, "Train your own small model".

Code, data and every eval run: [joshuatagoe/league-toxicity-slm @ `{code_commit}`](https://github.com/joshuatagoe/league-toxicity-slm/tree/{code_commit}).

## Behaviour spec

> Given one League of Legends chat line, the model replies with exactly one word, `toxic` or `nontoxic`. It says `toxic` only when the line attacks, mocks or demeans a player (insults, including disguised or non-English ones; sarcasm or taunts aimed at someone; "diff" put-downs), and `nontoxic` for game talk, callouts, self-criticism, quoting or reporting someone, swearing at the game, friendly banter, and telling a flamer to stop.

A reply passes if it is exactly one of the two words and matches the line's label.

## Results

`E1000` is the first 1,000 messages of the L2DTnH test split, scored against the expert annotators' labels. The pressure set is 40 hard lines (4 per rule) labelled under the project's written rules. The bar, set before any run, is precision ≥ 0.80 and F1 ≥ 0.75 on `E1000`.

| Model | Precision | Recall | F1 | Pressure set |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B, untrained | 0.466 | 0.471 | 0.468 | 25/40 |
| **This adapter (v2)** | **0.857** | 0.588 | 0.698 | **30/40** |
| v1 (expert-labelled data only) | 0.823 | 0.637 | 0.718 | 27/40 |
| Claude Opus 5.5, best prompt | 0.624 | 0.765 | 0.687 | 38/40 |
| GPT-5.6 Sol, best prompt | 0.419 | 0.882 | 0.568 | 34/40 |

The adapter clears the precision bar and misses the F1 bar. Base and tuned models run in bf16.

## Run the eval

From a checkout of the code commit above, in the training environment (`requirements-train.txt`):

```
python eval.py --model {repo_id} --revision <this repo's commit> --eval-set data/eval/E1000.jsonl
python eval.py --model {repo_id} --revision <this repo's commit> --eval-set data/pressure/PRESSURE.jsonl
```

Each command loads the base and this adapter, labels every message and writes per-message results to `runs/`. Use `--model Qwen/Qwen3-1.7B` for the untrained base.

The prompt is a fixed system message plus the chat line, rendered with Qwen3's chat template and thinking off; see `src/slm_prompt.py` and `src/slm.py`.

## Training

- **Method:** QLoRA on a 4-bit NF4 base. LoRA rank 16, alpha 16, dropout 0, on every attention and MLP projection. Learning rate 2e-4, cosine schedule, 1 epoch, batch 16, max length 256, seed 3407. Loss on the answer only. 725 steps, 97 minutes on one consumer GPU.
- **Data (11,595 rows, 11% toxic):**
  - 10,919 L2DTnH training messages with the experts' labels.
  - 676 lines written by Claude Opus 5.5, aimed at the failures v1 showed: sarcasm, taunts, "diff" put-downs, polite put-downs and disguised insults (toxic), and harmless lines with hostile-looking words (not toxic). Each was deduplicated, checked against every eval message for overlap, and checked by GPT-5.6 Sol against the written rules; 40 lines it disputed were left out.

## Limitations

- It still misses most hostility without insult words: it catches 4 of the 11 toxic pressure lines, and says `nontoxic` to "gg ez" and "wow great gank jungle, really helpful".
- It judges one line with no game state or earlier chat.
- English only, trained on chat from the 2026 L2DTnH release.
- For research. Don't use it to punish real players.

## Data and licence

Training and eval messages come from [L2DTnH](https://arxiv.org/abs/2604.10175) (Ave, Pekaric, Frohner, Apruzzese, 2026), released under the MIT licence. The adapter follows the base model's Apache 2.0 licence.
