"""Build the fine-tuning data (v1) from the experts' train split.

Usage: python -I scripts/make_sft_data.py

Writes data/sft/train_v1.jsonl and data/sft/val.jsonl, one row per message:
{"id", "label", "prompt": [system, user], "completion": [assistant]}.
The validation set is a stratified 10% of the train split; test rows are never used.
train_v1.jsonl is shuffled once with a fixed seed, so its first N rows are the
nested subsets for the data-efficiency curve (N, N/2, N/4, N/8).
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from slm_prompt import LABELS, messages  # noqa: E402

SEED = 7
VAL_SHARE = 0.10
TRAIN = ROOT / "data" / "raw" / "2_16000_chatlogs_english_only_train_split.csv"
OUT = ROOT / "data" / "sft"


def row(r: dict) -> dict:
    label = int(float(r["label"]))
    return {
        "id": r["id"],
        "label": label,
        "prompt": messages(r["message"].strip()),
        "completion": [{"role": "assistant", "content": LABELS[label]}],
    }


with TRAIN.open(newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["message"].strip()]
test_ids = {json.loads(l)["id"] for l in (ROOT / "data" / "eval" / "FULL.jsonl").open(encoding="utf-8")}
assert not test_ids & {r["id"] for r in rows}, "train rows overlap the test split"

rng = random.Random(SEED)
train, val = [], []
for label in (0, 1):
    group = [r for r in rows if int(float(r["label"])) == label]
    rng.shuffle(group)
    cut = round(len(group) * VAL_SHARE)
    val += group[:cut]
    train += group[cut:]
rng.shuffle(train)
rng.shuffle(val)

OUT.mkdir(parents=True, exist_ok=True)
hashes = {}
for name, data in (("train_v1.jsonl", train), ("val.jsonl", val)):
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in data:
            f.write(json.dumps(row(r), ensure_ascii=False) + "\n")
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    toxic = sum(int(float(r["label"])) for r in data)
    print(f"{name}: {len(data)} rows, {toxic} toxic ({toxic / len(data):.1%})")

(OUT / "manifest.json").write_text(json.dumps({"seed": SEED, "val_share": VAL_SHARE, "sha256": hashes}, indent=2) + "\n")
