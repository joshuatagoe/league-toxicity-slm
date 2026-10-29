"""Regrade existing runs after an eval set's labels change, without calling any model.

Usage: python -I scripts/regrade.py --eval-set PRESSURE

Each run keeps its predictions; only the gold label per message, the stored eval
hash and the scores are refreshed from the current eval file.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import confusion, scores  # noqa: E402
from run_ceiling import EVAL_SETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, choices=tuple(EVAL_SETS))
    args = ap.parse_args()
    path = EVAL_SETS[args.eval_set]
    gold = {r["id"]: r["label"] for r in (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for meta_path in sorted((ROOT / "runs").glob(f"*__{args.eval_set}.meta.json")):
        meta = json.loads(meta_path.read_text())
        run_path = ROOT / "runs" / f"{meta['cell']}.jsonl"
        records = [json.loads(l) for l in run_path.open(encoding="utf-8") if l.strip()]
        records = [r for r in records if r["id"] in gold]
        changed = sum(r["gold"] != gold[r["id"]] for r in records)
        for r in records:
            r["gold"] = gold[r["id"]]
        with run_path.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta.update(eval_sha256=sha, confusion=confusion(records), scores=scores(confusion(records)),
                    regraded_at=datetime.now(timezone.utc).isoformat())
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{meta['cell']}: {changed} gold labels changed, {len(records)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
