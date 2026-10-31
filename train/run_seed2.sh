#!/usr/bin/env bash
# Second training seed for the data-efficiency curve and for v2: same rows, same settings, seed 42 instead of 3407.
# The seed changes the LoRA initialisation and the order batches are drawn in, so the spread between
# the two seeds is training noise, which the eval-sampling intervals don't cover.
# Usage (from the repo root): bash train/run_seed2.sh
set -euo pipefail
PY=.venv-train/Scripts/python
BASE=Qwen/Qwen3-1.7B
SEED=42

score() {  # score <model> <run name>
  "$PY" -u eval.py --model "$1" --name "$2" --eval-set data/eval/E1000.jsonl
  "$PY" -u eval.py --model "$1" --name "$2" --eval-set data/pressure/PRESSURE.jsonl
}

for n in 1364 2729 5459 10919; do
  name=qwen3-1.7b-v1-n$n-seed$SEED
  echo "[$(date +%T)] training $name"
  "$PY" -u train/train_qlora.py --base-model "$BASE" --n "$n" --seed "$SEED" --out "models/$name"
  score "models/$name" "$name"
done

name=qwen3-1.7b-v2-seed$SEED
echo "[$(date +%T)] training $name"
"$PY" -u train/train_qlora.py --base-model "$BASE" --train-file data/sft/train_v2.jsonl --seed "$SEED" --out "models/$name"
score "models/$name" "$name"
echo "[$(date +%T)] all done"
