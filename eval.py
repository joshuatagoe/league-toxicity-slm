"""One-command evaluation of the small model, base or tuned.

Usage (training environment):
  .venv-train/Scripts/python eval.py --model Qwen/Qwen3-1.7B --eval-set data/eval/E1000.jsonl
  .venv-train/Scripts/python eval.py --model models/qwen3-1.7b-v1 --eval-set data/eval/E1000.jsonl
  .venv-train/Scripts/python eval.py --model <hf-user>/<repo> --revision <commit> --eval-set data/eval/FULL.jsonl

--model is a full model or a LoRA adapter (local folder or Hugging Face repo). Writes
runs/<name>__slm__na__<eval set>.jsonl and .meta.json in the same format as the
frontier runs, so src/report.py puts base, tuned and frontier models in one table.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from labels import parse_label  # noqa: E402
from metrics import confusion, scores  # noqa: E402
from slm import label, load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hub id, local folder, or LoRA adapter")
    ap.add_argument("--revision", help="pin a hub commit")
    ap.add_argument("--eval-set", required=True, help="path to an eval JSONL, e.g. data/eval/E1000.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", help="run name (default: derived from --model)")
    args = ap.parse_args()

    eval_path = Path(args.eval_set)
    rows = [json.loads(l) for l in eval_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    eval_name = eval_path.stem

    tokenizer, model = load(args.model, args.revision)
    start = time.perf_counter()
    raws = label(tokenizer, model, [r["message"] for r in rows], args.batch)
    per_message_ms = round((time.perf_counter() - start) * 1000 / max(1, len(rows)), 1)

    name = args.name or args.model.replace("\\", "/").rstrip("/").split("/")[-1]
    cell = f"{name}__slm__na__{eval_name}" + (f"__n{args.limit}" if args.limit else "")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    records = []
    with (runs / f"{cell}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r, raw in zip(rows, raws):
            pred = parse_label(raw)
            rec = {
                "id": r["id"], "message": r["message"], "gold": r["label"], "pred": pred,
                "status": "ok" if pred is not None else "parse_fail", "raw": raw, "stop_reason": None,
                "input_tokens": 0, "output_tokens": 0, "latency_ms": per_message_ms, "error": None,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    c = confusion(records)
    s = scores(c)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "cell": cell, "model": name, "model_ref": args.model, "revision": args.revision,
        "strategy": "slm", "effort": "na", "eval_set": eval_name, "limit": args.limit,
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "git_commit": commit, "finished_at": datetime.now(timezone.utc).isoformat(),
        "per_message_ms": per_message_ms, "confusion": c, "scores": s,
    }
    (runs / f"{cell}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{cell}: precision {s['precision']:.3f}  recall {s['recall']:.3f}  f1 {s['f1']:.3f}  "
          f"accuracy {s['accuracy']:.3f}  n {s['n']}  failures {c['failures']}  {per_message_ms} ms/message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
