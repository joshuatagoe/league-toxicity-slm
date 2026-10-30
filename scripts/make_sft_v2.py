"""Build train_v2: the v1 rows plus the gated teacher rows, shuffled with a fixed seed.

Usage: python -I scripts/make_sft_v2.py [--smoke]

The only change from v1 is data: 676 teacher-written lines aimed at the two failures
v1 showed on the pressure set. Same prompt, same format, same training settings.
Also writes train_v2_small.jsonl: the first 2,729 v1 rows (the same subset as the
v1 n2729 run) plus the same teacher rows, to test whether a small curated set holds.
--smoke instead mixes the first 240 v1 rows with data/smoke/teacher_v2.jsonl into
data/smoke/train_smoke.jsonl (see train/smoke.sh).
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

SMALL_V1_ROWS = 2729
SMOKE = "--smoke" in sys.argv
OUT = ROOT / "data" / "smoke" if SMOKE else SFT

v1 = [json.loads(l) for l in (SFT / "train_v1.jsonl").open(encoding="utf-8") if l.strip()]
teacher = [json.loads(l) for l in (OUT / "teacher_v2.jsonl").open(encoding="utf-8") if l.strip()]
teacher_rows = [
    {"id": f"t2-{i:04d}", "label": t["label"], "category": t["category"],
     "prompt": messages(t["message"]), "completion": [{"role": "assistant", "content": LABELS[t["label"]]}]}
    for i, t in enumerate(teacher)
]
manifest = json.loads((SFT / "manifest.json").read_text())
builds = [("train_smoke.jsonl", v1[:240])] if SMOKE else [("train_v2.jsonl", v1), ("train_v2_small.jsonl", v1[:SMALL_V1_ROWS])]
for name, base in builds:
    rows = base + teacher_rows
    random.Random(SEED).shuffle(rows)
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest[name.removesuffix(".jsonl")] = {"seed": SEED, "v1_rows": len(base), "teacher_rows": len(teacher_rows)}
    toxic = sum(r["label"] for r in rows)
    print(f"{name}: {len(rows)} rows ({len(base)} v1 + {len(teacher_rows)} teacher), {toxic} toxic ({toxic / len(rows):.1%})")
manifest.pop("v2", None)
if not SMOKE:
    (SFT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
