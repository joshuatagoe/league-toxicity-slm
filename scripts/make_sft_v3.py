"""Build train_v3: v2's data with a heavier dose of the teacher lines.

Usage: python -I scripts/make_sft_v3.py

v2 mixed the 676 teacher lines in once, 5.8% of its rows. It learned "diff" put-downs but not
sarcasm. v3 repeats each of those same teacher rows 4 times (2,704 rows, 19.8%) next to the same
10,919 expert rows; prompt, format and training settings stay as in v2. It tests the Brainlift's
open question: does a heavier dose teach sarcasm without costing precision?
Decided before training: v2 stays the shipped model whatever v3 scores, because choosing a model
by the 40-line pressure set would fit those 40 lines.
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "data" / "sft"
SEED = 11  # the same shuffle seed as v2
REPEAT = 4

v1 = [json.loads(l) for l in (SFT / "train_v1.jsonl").open(encoding="utf-8") if l.strip()]
teacher = [r for r in (json.loads(l) for l in (SFT / "train_v2.jsonl").open(encoding="utf-8") if l.strip()) if r["id"].startswith("t2-")]
assert len(teacher) == 676, len(teacher)
rows = v1 + [{**r, "id": f"{r['id']}-r{k}"} for k in range(REPEAT) for r in teacher]
random.Random(SEED).shuffle(rows)
path = SFT / "train_v3.jsonl"
with path.open("w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

manifest = json.loads((SFT / "manifest.json").read_text())
manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest["train_v3"] = {"seed": SEED, "v1_rows": len(v1), "teacher_rows": len(teacher), "teacher_repeat": REPEAT}
(SFT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
toxic = sum(r["label"] for r in rows)
print(f"{path.name}: {len(rows)} rows ({len(v1)} v1 + {len(teacher)} teacher x {REPEAT}), {toxic} toxic ({toxic / len(rows):.1%})")
