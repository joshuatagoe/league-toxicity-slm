"""Build train_v2: the v1 rows plus the gated teacher rows, shuffled with a fixed seed.

Usage: python -I scripts/make_sft_v2.py

The only change from v1 is data: 676 teacher-written lines aimed at the two failures
v1 showed on the pressure set. Same prompt, same format, same training settings.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from slm_prompt import LABELS, messages  # noqa: E402

SEED = 11
SFT = ROOT / "data" / "sft"

v1 = [json.loads(l) for l in (SFT / "train_v1.jsonl").open(encoding="utf-8") if l.strip()]
teacher = [json.loads(l) for l in (SFT / "teacher_v2.jsonl").open(encoding="utf-8") if l.strip()]
rows = v1 + [
    {"id": f"t2-{i:04d}", "label": t["label"], "category": t["category"],
     "prompt": messages(t["message"]), "completion": [{"role": "assistant", "content": LABELS[t["label"]]}]}
    for i, t in enumerate(teacher)
]
random.Random(SEED).shuffle(rows)

path = SFT / "train_v2.jsonl"
with path.open("w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
manifest = json.loads((SFT / "manifest.json").read_text())
manifest["sha256"]["train_v2.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest["v2"] = {"seed": SEED, "v1_rows": len(v1), "teacher_rows": len(teacher)}
(SFT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
toxic = sum(r["label"] for r in rows)
print(f"train_v2.jsonl: {len(rows)} rows ({len(v1)} v1 + {len(teacher)} teacher), {toxic} toxic ({toxic / len(rows):.1%})")
