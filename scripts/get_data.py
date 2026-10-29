"""Download L2DTnH at the pinned commit, check it, and freeze the eval sets.

Usage: python -I scripts/get_data.py

Writes data/raw/*.csv (not committed), data/eval/E1000.jsonl (the benchmark: the first
1,000 test rows) and data/eval/FULL.jsonl (all 3,035), and data/manifest.json with the SHA-256 of every file. If data/manifest.json already
exists, the raw files must match the hashes recorded there.
"""
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "66929d13e455d529f99feee2819374a60ecd6ad9"
BASE = f"https://raw.githubusercontent.com/irdin-pekaric/esorics26_toxicity/{PINNED_COMMIT}/model_creation/1_dataset/"
FILES = {
    "train": "2_16000_chatlogs_english_only_train_split.csv",
    "test": "2_16000_chatlogs_english_only_test_split.csv",
}
# Counts from the paper (arXiv:2604.10175): test split 2,755 non-toxic / 280 toxic.
# The paper's GPT-4o subset (667 / 40) can't be rebuilt from this file, so the
# benchmark is simply the first 1,000 test rows.
EXPECTED_TEST = (2755, 280)
BENCHMARK_ROWS = 1000
KEEP = ["id", "message", "label", "chatlog_id", "time", "champion_name"]

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EVAL = ROOT / "data" / "eval"
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(float(r["label"]))
    return rows


def counts(rows: list[dict]) -> tuple[int, int]:
    toxic = sum(r["label"] for r in rows)
    return len(rows) - toxic, toxic


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in KEEP}, ensure_ascii=False) + "\n")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = RAW / name
        if not dest.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE + name, dest)

    recorded = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    hashes = {name: sha256(RAW / name) for name in FILES.values()}
    if recorded and recorded["raw"] != hashes:
        print("raw files do not match data/manifest.json; delete data/raw and re-run")
        return 1

    train = read_rows(RAW / FILES["train"])
    test = read_rows(RAW / FILES["test"])
    benchmark = test[:BENCHMARK_ROWS]
    print(f"train: {counts(train)} | test: {counts(test)} | E1000: {counts(benchmark)}  (non-toxic, toxic)")

    ok = True
    if counts(test) != EXPECTED_TEST:
        print(f"test counts differ from the paper: expected {EXPECTED_TEST}")
        ok = False
    test_ids = {r["id"] for r in test}
    overlap = sum(r["id"] in test_ids for r in train)
    if overlap:
        print(f"{overlap} train rows share an id with the test split")
        ok = False
    if not ok:
        return 1

    for stale in EVAL.glob("*.jsonl"):
        stale.unlink()
    write_jsonl(EVAL / "E1000.jsonl", benchmark)
    write_jsonl(EVAL / "FULL.jsonl", test)
    manifest = {
        "source": f"irdin-pekaric/esorics26_toxicity@{PINNED_COMMIT}",
        "raw": hashes,
        "eval": {p.name: sha256(p) for p in sorted(EVAL.glob("*.jsonl"))},
        "counts": {"train": counts(train), "test": counts(test), "E1000": counts(benchmark)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("eval sets frozen; hashes in data/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
