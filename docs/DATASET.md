# Dataset description

Every file the model was trained or scored on: its shape, where it came from and who labelled it. Hashes are in [data/manifest.json](../data/manifest.json) and [data/sft/manifest.json](../data/sft/manifest.json).

## At a glance

| File | Rows | Toxic | Labels from | Used for |
| --- | --- | --- | --- | --- |
| `data/sft/train_v1.jsonl` | 10,919 | 1,006 (9.2%) | L2DTnH experts | Training v1, and the data-efficiency curve (its first N rows) |
| `data/sft/teacher_v2.jsonl` | 676 | 269 (39.8%) | The category it was written for, under the written rules | Mixed into v2 and v2-small |
| `data/sft/train_v2.jsonl` | 11,595 | 1,275 (11.0%) | 10,919 expert + 676 teacher | Training v2, the shipped model |
| `data/sft/train_v2_small.jsonl` | 3,405 | 501 (14.7%) | 2,729 expert + 676 teacher | Training v2-small |
| `data/sft/val.jsonl` | 1,213 | 112 (9.2%) | L2DTnH experts | Held aside; not used for training or any reported score |
| `data/eval/E1000.jsonl` | 1,000 | 102 (10.2%) | L2DTnH experts | The benchmark: every precision, recall and F1 in the reports |
| `data/eval/FULL.jsonl` | 3,035 | 280 (9.2%) | L2DTnH experts | The whole test split; confirmed the best prompted cell |
| `data/pressure/PRESSURE.jsonl` | 40 | 11 (27.5%) | The project author, under [LABELING_RULES.md](LABELING_RULES.md) | Robustness: 4 hard lines for each of 10 kinds |
| `data/sft/teacher_v2_disputed.jsonl` | 40 | — | Teacher and critic disagreed | Left out of training |
| `data/smoke/*` | 20 teacher + 260 mixed | — | As above | The smoke test only (`train/smoke.sh`) |

**Expert-labelled vs teacher-written.** v1 trained only on expert-labelled real chat. v2 adds 676 teacher-written lines, 5.8% of its data. Teacher rows have ids starting `t2-` and a `category` field; expert rows have the L2DTnH message id. Every eval set is real chat, except the pressure set, which the author wrote and labelled.

## Where the real chat comes from

[L2DTnH](https://arxiv.org/abs/2604.10175) (Ave, Pekaric, Frohner, Apruzzese, 2026) is League of Legends chat from reported games, released under the MIT licence at [irdin-pekaric/esorics26_toxicity](https://github.com/irdin-pekaric/esorics26_toxicity). This project pins commit `66929d13e455d529f99feee2819374a60ecd6ad9`. The raw CSVs are not committed; `scripts/get_data.py` downloads them at that commit and checks their hashes and row counts.

- **Raw files:** a train split of 12,136 messages (1,118 toxic) and a test split of 3,035 (280 toxic). Each row has the message, the speaker's relation to the reported player (offender, ally or enemy), a timestamp, report counts and reasons, the chat log id, the champion and the label. Two more columns, `old_label` and `wrongly classified`, show a second review pass that the paper doesn't describe. It covered 1,526 messages across both splits and changed 630 labels: 365 toxic to fine and 265 fine to toxic.
- **How the experts labelled, per the paper:**
  - Eight players with 6 to 20 years of League experience labelled the first 5,000 messages independently. A message was toxic if at least 2 of them flagged it (Fleiss' κ 0.62; 4,704 of the 5,000 were unanimous).
  - Three of them labelled the rest, discussing disagreements until they agreed.
  - They could see the surrounding chat; the model sees one line.
- **What this project built from it** (`scripts/make_sft_data.py`, seed 7):
  - **Train split:** 4 rows with empty messages were dropped. The other 12,132 were split, stratified by label, into `train_v1.jsonl` (90%) and `val.jsonl` (10%). `train_v1.jsonl` is shuffled once, so its first 1,364, 2,729, 5,459 and 10,919 rows are nested subsets.
  - **Overlap check:** the script asserts that no training id appears in the test split.
  - **Test split:** `FULL.jsonl` is the whole test split. `E1000.jsonl` is its first 1,000 rows (898 fine, 102 toxic), the benchmark used everywhere.

## Where the teacher lines come from

`scripts/gen_teacher_v2.py` asked Claude Opus 5.5 for League chat lines in 12 categories, aimed at the two failures v1 showed on the pressure set. Each category fixes the label, under the written rules:

| Toxic categories (269 kept) | Not-toxic categories (407 kept) |
| --- | --- |
| sarcasm at a teammate (57), taunting the enemy (48), "diff" put-downs (56), polite put-downs (48), disguised insults (60) | quoting or reporting (59), game complaints (60), self-criticism (59), harmless trigger words (59), feed callouts (60), friendly banter (50), telling a flamer to stop (60) |

**The gates.** Of 720 generated lines:
- None were duplicates or outside 2–120 characters.
- 4 were dropped by the leakage check: 2 matched an eval message and 2 were within 0.8 similarity of a pressure line.
- A GPT-5.6 Sol critic labelled the remaining 716 under [LABELING_RULES.md](LABELING_RULES.md). It disputed 40, which were left out (`teacher_v2_disputed.jsonl`).

The teacher calls and the critic cost $0.81 at list prices (`data/sft/teacher_v2_meta.json`). `scripts/make_sft_v2.py` (seed 11) mixes the kept lines into the expert rows.

## The author's labels

- **Pressure set** (`data/pressure/PRESSURE.jsonl`): 40 lines the author wrote to trip models, 4 for each of 10 kinds, labelled under the written rules: sarcasm, quoting or reporting, self-criticism, talk about the game, harmless trigger words, banter, blunt callouts, taunts, blame without an insult, and disguised insults. No teacher line is within 0.8 similarity of any of them.
- **Relabel study** (`relabel/`): 140 messages the author relabelled blind, with the surrounding chat, to see whose standard the prompted models' errors reflect ([results/relabel_analysis.md](../results/relabel_analysis.md)). Not used for training or scoring.

## Formats

All files are JSON Lines, UTF-8, with LF line endings (the hashes depend on it).

```jsonc
// Training rows (train_v1, train_v2, train_v2_small, val): the exact prompt the model sees
{"id": "15922", "label": 0,
 "prompt": [{"role": "system", "content": "You moderate League of Legends chat. ... Reply with exactly one word: toxic or nontoxic."},
            {"role": "user", "content": "place"}],
 "completion": [{"role": "assistant", "content": "nontoxic"}]}
// Teacher rows also carry "category"; their ids start with "t2-".

// Eval rows (E1000, FULL)
{"id": "9524", "message": "this mid", "label": 0, "chatlog_id": "1005.0", "time": "00:16:56", "champion_name": "Jarvan IV"}

// Pressure rows
{"id": "p-01", "message": "wow great gank jungle, really helpful", "label": 1, "kind": "sarcasm at a teammate"}
```

`label` is 1 for toxic and 0 for not toxic throughout.

## Known limits

- **The two label sources disagree on some lines.** The experts labelled many bare feed callouts toxic, and borderline lines lean toxic under the 2-of-8 rule. The written rules lean not toxic when context is missing. See [LABELING_RULES.md](LABELING_RULES.md#where-these-differ-from-the-l2dtnh-experts).
- **One line, no context.** The experts saw the chat around each message; the model doesn't. Some expert labels can't be recovered from the line alone (`GAREN TANK`, `19 assist`).
- **Toxic lines are rare:** about 9% of real chat. Recall on 102 toxic benchmark lines has wide intervals.
- **The teacher lines come from one model.** Their idea of sarcasm or a taunt is Claude's, which may differ from how players write.
- **English only, and only from reported games.** The most common report reasons are verbal abuse, negative attitude, offensive language and assisting the enemy team, so the chat skews heated.
- **These are real players' messages.** Use them for research only, and never to label a real person as toxic.

For the final submission the training and eval files will be published as a Hugging Face dataset, with this page as its card.
