#!/usr/bin/env bash
# Rescore the untrained and every tuned v1 model in bf16 on both eval sets.
# Needed because the first scores used a 4-bit base, which broke the untrained model (see docs/PLAN.md).
# Usage (from the repo root): bash train/rescore_bf16.sh
set -euo pipefail
PY=.venv-train/Scripts/python
score() {  # score <model> <run name>
  "$PY" -u eval.py --model "$1" --name "$2" --eval-set data/eval/E1000.jsonl
  "$PY" -u eval.py --model "$1" --name "$2" --eval-set data/pressure/PRESSURE.jsonl
}
score Qwen/Qwen3-1.7B qwen3-1.7b-base
for n in 10919 1364 2729; do  # n5459 was already scored in bf16 by run_v1.sh
  score "models/qwen3-1.7b-v1-n$n" "qwen3-1.7b-v1-n$n"
done
