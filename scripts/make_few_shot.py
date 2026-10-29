"""Draw the few-shot examples from the train split with a fixed seed.

Usage: python -I scripts/make_few_shot.py
Writes prompts/few_shot_examples.json: 6 toxic and 6 non-toxic messages, shuffled.
Test rows are never used; the examples are drawn at random, not hand-picked.
"""
import csv
import json
import random
from pathlib import Path

SEED = 1234
PER_CLASS = 6
MIN_CHARS = 6  # skip one-word replies like "gj" or "y", which teach nothing

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "raw" / "2_16000_chatlogs_english_only_train_split.csv"

with TRAIN.open(newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if len(r["message"].strip()) >= MIN_CHARS]

rng = random.Random(SEED)
picked = []
for label, name in ((1, "toxic"), (0, "nontoxic")):
    pool = [r for r in rows if int(float(r["label"])) == label]
    picked += [{"id": r["id"], "message": r["message"].strip(), "label": name} for r in rng.sample(pool, PER_CLASS)]
rng.shuffle(picked)

out = ROOT / "prompts" / "few_shot_examples.json"
out.write_text(json.dumps(picked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for e in picked:
    print(f"{e['label']:>9}  {e['message']}")
