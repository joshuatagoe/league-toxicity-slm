#!/usr/bin/env bash
# Version 2: same settings as v1, plus the 676 gated teacher rows. The only change is data.
# Usage (from the repo root): bash train/run_v2.sh
set -euo pipefail
PY=.venv-train/Scripts/python
OUT=models/qwen3-1.7b-v2
"$PY" -u train/train_qlora.py --base-model Qwen/Qwen3-1.7B --train-file data/sft/train_v2.jsonl --out "$OUT"
"$PY" -u eval.py --model "$OUT" --name qwen3-1.7b-v2 --eval-set data/eval/E1000.jsonl
"$PY" -u eval.py --model "$OUT" --name qwen3-1.7b-v2 --eval-set data/pressure/PRESSURE.jsonl

# v2-small: the same 2,729 real rows as the v1 n2729 run, plus the same teacher rows.
SMALL=models/qwen3-1.7b-v2-small
"$PY" -u train/train_qlora.py --base-model Qwen/Qwen3-1.7B --train-file data/sft/train_v2_small.jsonl --out "$SMALL"
"$PY" -u eval.py --model "$SMALL" --name qwen3-1.7b-v2-small --eval-set data/eval/E1000.jsonl
"$PY" -u eval.py --model "$SMALL" --name qwen3-1.7b-v2-small --eval-set data/pressure/PRESSURE.jsonl
