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
| Training and eval data (public) | [Juchen/league-toxicity-chat](https://huggingface.co/datasets/Juchen/league-toxicity-chat) at commit `857733be37fe6e278e416e96c7341880f8754046`, described in [docs/DATASET.md](docs/DATASET.md) |
| Eval code | this repo at commit `3745c5e9894ea5b6d5d57b6e945fd990ffd29154` |

One command per eval set, in the training environment (`requirements-train.txt`):

```
python eval.py --model Juchen/league-toxicity-qwen3-1.7b --revision 80e3d1bac9505024f49cf3439926f2be7d2e809d --eval-set data/eval/E1000.jsonl
```

Use `--eval-set data/pressure/PRESSURE.jsonl` for the pressure set, or `--model Qwen/Qwen3-1.7B` for the untrained base. Run from the Hub this way, the adapter gave the same answer as the local run on all 1,000 `E1000` messages and all 40 pressure lines (`runs/qwen3-1.7b-v2-hub__*`).

[manifest.json](manifest.json) records the student, teacher and critic, every seed, the SHA-256 of every data file, the library versions and every model's scores (`scripts/make_manifest.py`).

### Base vs tuned on a new set, in one command

For the staff held-out set, or any file of chat lines:

```
.venv-train/Scripts/python compare.py --eval-set path/to/heldout.csv
```

It runs `eval.py` on the untrained base and on the public v2 adapter, both at the commits above. Then it prints each model's precision, recall, F1 and accuracy, the paired difference with a 95% interval, and every line where the two models disagree. The per-line answers go to `runs/`.

- **Formats:** JSON Lines, JSON, CSV, TSV, or plain text with one chat line per row.
- **Message field:** the first of `message`, `text`, `input`, `prompt` and similar names, or the last user turn of a `messages` list.
- **Label field:** the first of `label`, `gold`, `expected`, `target` or `toxic`, written as 1/0, true/false or toxic/nontoxic.
- **Missing or unreadable labels:** a row with no label is answered but not scored. A label it can't read stops the run instead of being guessed.

The reader is `src/evalsets.py`, tested in `tests/test_evalsets.py`.

### Rerun one point of each ablation

- **Prompt ceiling (Ablation 1).** The command below scores Claude's best cell on the first 100 benchmark lines. It needs `ANTHROPIC_API_KEY` in `.env` and costs about $0.10.

  ```
  python -I src/run_ceiling.py --model claude-opus-5-5 --strategy zero_shot_p1 --eval-set E1000 --limit 100
  ```

  It writes a new `runs/claude-opus-5-5__zero_shot_p1__low__E1000__n100.jsonl`; compare it with the first 100 lines of the full run. Claude doesn't accept a temperature, so a few borderline answers can differ between runs. `--model baseline:keyword` is free and reproduces exactly.
- **Data-efficiency curve (Ablation 2).** On a CUDA GPU, the first command trains the smallest point and the next two score it:

  ```
  .venv-train/Scripts/python train/train_qlora.py --n 1364 --seed 3407 --out models/rerun-n1364
  .venv-train/Scripts/python eval.py --model models/rerun-n1364 --eval-set data/eval/E1000.jsonl
  .venv-train/Scripts/python eval.py --model models/rerun-n1364 --eval-set data/pressure/PRESSURE.jsonl
  ```

  Compare with `results/train_logs/qwen3-1.7b-v1-n1364.json` and `runs/qwen3-1.7b-v1-n1364__*`. GPU training isn't bit-exact, so expect small differences; the two seeds per size in [results/slm_results.md](results/slm_results.md#data-efficiency-curve) show how large training noise is.

## Demo

**Project page: [claude.ai/artifact/Em9KAaiwVMFHCDZzaGMC2o](https://claude.ai/artifact/Em9KAaiwVMFHCDZzaGMC2o)**, the headline results, the live demo and every link below on one page, above the cohort labelling study.

**Live, in Colab: [demo/colab_demo.ipynb](https://colab.research.google.com/github/joshuatagoe/league-toxicity-slm/blob/main/demo/colab_demo.ipynb).** Choose the free T4 GPU and Run all, then type any chat line. The untrained Qwen3-1.7B and the public v2 adapter, both at the commits above, label it side by side. The notebook also labels the demo page's example lines next to their labels, and checks itself against the recorded eval runs on the 40 pressure lines. It needs a Google account, and the first run takes a few minutes to install three libraries and download about 4 GB.
- **On a T4 the weights load in fp16**, because a T4 has no bf16. On an L4 or A100 they load in bf16, as in the eval runs.
- **Tested by running the notebook's own code cells on this laptop's GPU.** In bf16 both models gave the same answer as their eval runs on all 1,000 benchmark lines and all 40 pressure lines.
- **In fp16, standing in for a T4, the tuned model matched on 998 of 1,000 and 40 of 40.** Its two differences are new false alarms (`fuck blue`, `This engage was stupid pant`), which would take its benchmark precision from 0.857 to 0.833. The untrained model matched on 994 and 39; the pressure line is `cry more top` again.
- **On Colab's T4 it gave the same results as that fp16 run** on the pressure lines: the tuned model matched on all 40 (30/40 right), the untrained one on 39. A typed line took under a second for both models. Colab ships an old `torchao` that peft 0.21 refuses to load beside, so the notebook uninstalls it first; the demo doesn't use it.

**Video: [youtu.be/9-evuZoQeQk](https://youtu.be/9-evuZoQeQk)**, a narrated walkthrough of the tuned model next to the untrained one (4:54; sections 4 and 5 were re-recorded for the final submission with the second-seed, v3 and sarcasm-probe results) (also in the repo as [docs/walkthrough-voiced.mp4](docs/walkthrough-voiced.mp4); the silent cut is [docs/walkthrough.mp4](docs/walkthrough.mp4) and the script is [docs/walkthrough-script.md](docs/walkthrough-script.md)). To run the page it records (needs a CUDA GPU and the training environment):

```
.venv-train/Scripts/python demo/server.py
```

Then open http://127.0.0.1:8780. The server loads Qwen3-1.7B once in bf16 and attaches the public v2 adapter at its pinned commit, plus v1 if `models/` has it. The untrained column is the same weights with the adapters switched off. Prompt, chat template and greedy decoding are the ones `eval.py` uses: the page's answers matched the eval runs on all 40 pressure lines and the 8 benchmark lines it shows. The Claude column is read from Claude's eval run, not called live.

One caveat: on borderline lines, the untrained model's answer can change with the other lines in its batch (bf16 with left padding). One benchmark line did this, so the example set uses lines whose answers held both in the set and alone.

**A Hugging Face Space, not published.** `scripts/hf_space.py` builds the same page as a Docker Space, with the untrained base and the public v2 adapter at their pinned commits. Hugging Face now needs a PRO account to run a Docker Space, even on its free CPU hardware, so the Colab notebook is the hosted demo instead. Run on this laptop's CPU in float32, the built page matched the eval runs on 108 of 110 answers; the two misses were the untrained model on `cry more top`, a borderline line.

`demo/video/` holds the recorder (`record.cjs`, Playwright) and the narration studio. To record the voice-over, run `python -I demo/video/studio.py --video docs/walkthrough.mp4 --sections demo/video/sections.json --script docs/walkthrough-script.md --clips demo/video/clips` and open http://127.0.0.1:8770.

## Reports

| Report | What it covers |
| --- | --- |
| [docs/BRAINLIFT.md](docs/BRAINLIFT.md) | The thesis and whether data → behaviour held |
| [docs/DATASET.md](docs/DATASET.md) | Every training and eval file: shape, source, who labelled it, and which rows are teacher-written |
| [docs/SLM_REPORT.md](docs/SLM_REPORT.md) | Base vs tuned, the v2 data fix and its heavier dose (v3), the two-seed data-efficiency curve and minimum N (Ablation 2), why scoring is exact match, the smoke test |
| [docs/CEILING_REPORT.md](docs/CEILING_REPORT.md) and [results/ablation1.md](results/ablation1.md) | The prompt-ceiling ablation (Ablation 1) and its failure mode |
| [results/slm_results.md](results/slm_results.md) | Generated tables behind the small-model report, with paired intervals (`scripts/slm_results.py`) |
| [results/sarcasm_probe.md](results/sarcasm_probe.md) | Why the tuned model misses sarcasm: whether it sees it, and what flagging it would cost (`src/run_sarcasm_probe.py`) |
| [results/train_logs/](results/train_logs/) | Every training run's settings, data hash, loss, steps and code commit |
| [manifest.json](manifest.json) | Student, teacher, seeds, data hashes, library versions and scores in one file |
| [results/smoke.log](results/smoke.log) | `bash train/smoke.sh`: teacher generation, training and eval on a small batch, end to end |

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
| Fine-tuned BERT, rescored here on `E1000` | 0.8471 | 0.7059 | 0.7701 |

Source: Table of results in [arXiv:2604.10175](https://arxiv.org/abs/2604.10175). The last row is ours.

The fine-tuned BERT's weights are in the paper's repository. `src/run_paper_bert.py` downloads them at the pinned commit (438 MB, into `models/paper-bert/`), checks each file's SHA-256 and scores an eval set on the CPU, writing `runs/` files like `eval.py`:

```
.venv-train/Scripts/python -I src/run_paper_bert.py --eval-set data/eval/E1000.jsonl
```

Its test-split numbers are flattered. The paper's training script kept the epoch with the lowest test-split loss and chose its 0.58 threshold on the test split, and `E1000` is the first 1,000 rows of that split. On the 40 pressure lines, which it never saw, it scores 25/40, the same as the untrained Qwen3-1.7B. The shipped v2 scores 30/40. The full comparison is in [results/slm_results.md](results/slm_results.md).
