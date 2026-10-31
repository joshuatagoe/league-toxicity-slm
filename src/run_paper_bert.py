"""Score the L2DTnH paper's fine-tuned BERT on an eval set, as a baseline next to the small model.

Usage (training environment, from the repo root):
  .venv-train/Scripts/python -I src/run_paper_bert.py --eval-set data/eval/E1000.jsonl
  .venv-train/Scripts/python -I src/run_paper_bert.py --eval-set data/pressure/PRESSURE.jsonl

The model is the paper authors' fine-tune of unitary/toxic-bert (Ave et al., ESORICS 2026), from
irdin-pekaric/esorics26_toxicity (MIT) at the commit the data is pinned to. The first run downloads
its six files (438 MB) into models/paper-bert/; every run checks their SHA-256 before loading.
Scoring follows the paper's script, model_creation/3_finetuning/3_1_finetuning_enlish_only/
finetune_english_only.py: truncate at 192 tokens, toxic when softmax(logits)[1] >= 0.58, the
threshold it chose. It runs on the CPU by default, so it can share the machine with training.

That script used the test split as its validation set: it kept the epoch with the lowest test loss
and chose 0.58 by sweeping thresholds on the test split. E1000 is the first 1,000 rows of that
split, so the model's E1000 score is flattered. The pressure set is new to it.

Writes runs/paper-bert__classifier__na__<eval set>.jsonl and .meta.json in eval.py's format; `raw`
holds the toxic probability.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evalsets import read_eval_set  # noqa: E402
from metrics import confusion, scores  # noqa: E402

NAME = "paper-bert"
REPO = "irdin-pekaric/esorics26_toxicity"
COMMIT = "66929d13e455d529f99feee2819374a60ecd6ad9"  # the commit scripts/get_data.py pins
FOLDER = "model_creation/3_finetuning/3_1_finetuning_enlish_only/output/finetuned_model"
FILES = {  # file: SHA-256; model.safetensors is a Git LFS file, and its LFS pointer names the same hash
    "config.json": "08046c7d386f2d8ce626b159fabd594790cf6ac74679bfdc5cec169322e1f364",
    "model.safetensors": "e4a01ae8c4bf431fb6514580a4dd111edca0d9033d473b054d4dc3d3305352af",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "1df3266d8f29dd3afdf73a9ee31322104ff93f8575a6bae998be07284b792e89",
    "tokenizer_config.json": "7b082b48a08a0b5c5939d99472e33addce3d96202155e33f7928cecd171186a0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MAX_LEN = 192
THRESHOLD = 0.58
TRAIN_ROWS = 12136  # "train_shape" in the paper's output/summary.json at COMMIT
DIR = ROOT / "models" / NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> None:
    """Download any missing file at the pinned commit, then check every file's hash."""
    DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = DIR / name
        if not path.exists():
            host = "media.githubusercontent.com/media" if name == "model.safetensors" else "raw.githubusercontent.com"
            print(f"downloading {name}")
            part = path.with_name(name + ".part")
            urllib.request.urlretrieve(f"https://{host}/{REPO}/{COMMIT}/{FOLDER}/{name}", part)
            part.replace(path)
        if sha256(path) != digest:
            sys.exit(f"{path.relative_to(ROOT)} doesn't match its pinned SHA-256; delete it and run again")


def toxic_probability(messages: list[str], batch: int, device: str) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DIR, use_safetensors=True).to(device).eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(messages), batch):
            enc = tokenizer(messages[i:i + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            probs += torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].tolist()
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, help="path to an eval set, e.g. data/eval/E1000.jsonl or a held-out CSV")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's choice)")
    args = ap.parse_args()

    eval_path = Path(args.eval_set)
    rows = read_eval_set(eval_path)
    if args.limit:
        rows = rows[: args.limit]
    eval_name = eval_path.stem

    fetch()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    start = time.perf_counter()
    probs = toxic_probability([r["message"] for r in rows], args.batch, args.device)
    per_message_ms = round((time.perf_counter() - start) * 1000 / max(1, len(rows)), 1)

    cell = f"{NAME}__classifier__na__{eval_name}" + (f"__n{args.limit}" if args.limit else "")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    records = []
    with (runs / f"{cell}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(rows, probs):
            rec = {
                "id": r["id"], "message": r["message"], "gold": r["label"], "pred": int(p >= THRESHOLD),
                "status": "ok", "raw": f"{p:.6f}", "stop_reason": None,
                "input_tokens": 0, "output_tokens": 0, "latency_ms": per_message_ms, "error": None,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    labelled = [r for r in records if r["gold"] is not None]
    c = confusion(labelled)
    s = scores(c) if labelled else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "cell": cell, "model": NAME, "model_ref": f"https://github.com/{REPO}/tree/{COMMIT}/{FOLDER}", "revision": COMMIT,
        "strategy": "classifier", "effort": "na", "eval_set": eval_name, "limit": args.limit,
        "quantization": "fp32", "device": args.device, "max_length": MAX_LEN, "threshold": THRESHOLD,
        "train_rows": TRAIN_ROWS, "weights_sha256": FILES["model.safetensors"],
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "git_commit": commit, "finished_at": datetime.now(timezone.utc).isoformat(),
        "per_message_ms": per_message_ms, "confusion": c, "scores": s,
    }
    (runs / f"{cell}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    unlabelled = len(records) - len(labelled)
    if s:
        print(f"{cell}: precision {s['precision']:.3f}  recall {s['recall']:.3f}  f1 {s['f1']:.3f}  "
              f"accuracy {s['accuracy']:.3f}  n {s['n']}  {per_message_ms} ms/message")
    if unlabelled:
        said = [r["pred"] for r in records if r["gold"] is None]
        print(f"{cell}: {unlabelled} rows without a label, not scored: {said.count(1)} toxic, {said.count(0)} nontoxic")
    print(f"answers: runs/{cell}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
