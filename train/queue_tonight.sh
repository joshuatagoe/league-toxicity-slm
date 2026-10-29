#!/usr/bin/env bash
# Wait for run_v1.sh to finish its last step, then rescore v1 in bf16 and train v2 and v2-small.
# Usage (from the repo root): bash train/queue_tonight.sh
set -euo pipefail
DONE=runs/qwen3-1.7b-v1-n5459__slm__na__PRESSURE.meta.json
for _ in $(seq 1 360); do  # up to 3 hours
  [ -f "$DONE" ] && break
  sleep 30
done
[ -f "$DONE" ] || { echo "run_v1.sh did not finish within 3 hours; nothing started"; exit 1; }
echo "[$(date +%T)] v1 sweep finished; rescoring in bf16"
bash train/rescore_bf16.sh
echo "[$(date +%T)] rescore done; training v2"
bash train/run_v2.sh
echo "[$(date +%T)] all done"
