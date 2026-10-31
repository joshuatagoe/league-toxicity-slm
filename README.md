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

**Video: [youtube.com/watch?v=yJ5kVQjnCCw](https://www.youtube.com/watch?v=yJ5kVQjnCCw)**, a narrated walkthrough of the tuned model next to the untrained one (also in the repo as [docs/walkthrough-voiced.mp4](docs/walkthrough-voiced.mp4); the silent cut is [docs/walkthrough.mp4](docs/walkthrough.mp4) and the script is [docs/walkthrough-script.md](docs/walkthrough-script.md)). To run the page it records (needs a CUDA GPU and the training environment):

```
.venv-train/Scripts/python demo/server.py
```

Then open http://127.0.0.1:8780. The server loads Qwen3-1.7B once in bf16 and attaches the public v2 adapter at its pinned commit, plus v1 if `models/` has it. The untrained column is the same weights with the adapters switched off. Prompt, chat template and greedy decoding are the ones `eval.py` uses: the page's answers matched the eval runs on all 40 pressure lines and the 8 benchmark lines it shows. The Claude column is read from Claude's eval run, not called live.

One caveat: on borderline lines, the untrained model's answer can change with the other lines in its batch (bf16 with left padding). One benchmark line did this, so the example set uses lines whose answers held both in the set and alone.

**Hosted version.** `scripts/hf_space.py` builds the same page as a Docker Space for Hugging Face's free CPU tier (2 vCPUs, 16 GB). It has the untrained base and the public v2 adapter at their pinned commits, with the weights baked into the image; there is no v1 column. Build it into `build/space` with `.venv-train/Scripts/python -I scripts/hf_space.py`; `--push` publishes it.
- **On a CPU the weights load in float32, not bf16.** Run that way on this laptop, the built server matched the eval runs on 108 of 110 answers. The test sent the page's three example sets as the page does, then the 40 pressure lines in two requests of 20.
- **v2 matched on all 55 lines.** The two misses are the untrained model answering `toxic` to `cry more top`, a borderline line whose answer also changes with batch size on the GPU.
- **Speed isn't measured yet.** Here each line took 3 to 12 s per model, with 2 threads and the model partly paged to disk for lack of memory. The Space has more memory, so its speed needs checking once it's published.

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
