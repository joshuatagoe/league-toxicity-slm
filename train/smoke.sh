#!/usr/bin/env bash
# The whole loop on a small batch: teacher generation -> training data -> QLoRA -> eval, base vs tuned.
# Uses Qwen3-0.6B and 16 training steps, so it finishes in a few minutes and costs a few cents of API calls.
# Usage (from the repo root): bash train/smoke.sh 2>&1 | tee results/smoke.log
set -euo pipefail
export TQDM_DISABLE=1
API=.venv/Scripts/python
PY=.venv-train/Scripts/python
BASE=Qwen/Qwen3-0.6B
OUT=models/smoke-qwen3-0.6b

echo "[$(date +%T)] 1/4 teacher writes lines for two categories, gated by dedupe, leakage and a critic"
"$API" -I scripts/gen_teacher_v2.py --smoke
echo "[$(date +%T)] 2/4 mix them with 240 expert-labelled rows"
"$API" -I scripts/make_sft_v2.py --smoke
echo "[$(date +%T)] 3/4 QLoRA, 16 steps"
"$PY" train/train_qlora.py --base-model "$BASE" --train-file data/smoke/train_smoke.jsonl --max-steps 16 --out "$OUT"
echo "[$(date +%T)] 4/4 score base and tuned on the first 50 benchmark lines and the pressure set"
score() {  # score <model> <run name>; --limit keeps these out of the full-run tables
  "$PY" eval.py --model "$1" --name "$2" --eval-set data/eval/E1000.jsonl --limit 50
  "$PY" eval.py --model "$1" --name "$2" --eval-set data/pressure/PRESSURE.jsonl --limit 40
}
score "$BASE" smoke-qwen3-0.6b-base
score "$OUT" smoke-qwen3-0.6b-tuned
echo "[$(date +%T)] done"
