#!/usr/bin/env bash
# Version 1: base vs tuned Qwen3-1.7B on the experts' labels, plus the data-efficiency sweep.
# Usage (from the repo root): bash train/run_v1.sh
set -euo pipefail
PY=.venv-train/Scripts/python
BASE=Qwen/Qwen3-1.7B
N=10919  # all of data/sft/train_v1.jsonl

score() {  # score <model> <run name>
  "$PY" eval.py --model "$1" --name "$2" --eval-set data/eval/E1000.jsonl
  "$PY" eval.py --model "$1" --name "$2" --eval-set data/pressure/PRESSURE.jsonl
}

score "$BASE" qwen3-1.7b-base

for n in $N $((N / 8)) $((N / 4)) $((N / 2)); do
  out=models/qwen3-1.7b-v1-n$n
  "$PY" train/train_qlora.py --base-model "$BASE" --n "$n" --out "$out"
  score "$out" "qwen3-1.7b-v1-n$n"
done
