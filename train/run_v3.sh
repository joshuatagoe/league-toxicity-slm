#!/usr/bin/env bash
# Version 3: v2's teacher lines at 4x the dose (scripts/make_sft_v3.py), same settings and seed as v2.
# Waits for train/run_seed2.sh if it is still using the GPU.
# Usage (from the repo root): bash train/run_v3.sh
set -euo pipefail
PY=.venv-train/Scripts/python
DONE=runs/qwen3-1.7b-v2-seed42__slm__na__PRESSURE.meta.json
if pgrep -f run_seed2.sh >/dev/null 2>&1 || { [ -f results/seed2.log ] && [ ! -f "$DONE" ]; }; then
  echo "[$(date +%T)] waiting for run_seed2.sh to finish"
  for _ in $(seq 1 720); do  # up to 6 hours
    [ -f "$DONE" ] && break
    sleep 30
  done
  [ -f "$DONE" ] || { echo "run_seed2.sh did not finish within 6 hours; nothing started"; exit 1; }
fi
OUT=models/qwen3-1.7b-v3
echo "[$(date +%T)] training qwen3-1.7b-v3"
"$PY" -u train/train_qlora.py --base-model Qwen/Qwen3-1.7B --train-file data/sft/train_v3.jsonl --out "$OUT"
"$PY" -u eval.py --model "$OUT" --name qwen3-1.7b-v3 --eval-set data/eval/E1000.jsonl
"$PY" -u eval.py --model "$OUT" --name qwen3-1.7b-v3 --eval-set data/pressure/PRESSURE.jsonl
echo "[$(date +%T)] all done"
