"""Write manifest.json: what the shipped model is, what it was trained on, with what, and how it scored.

Usage (training environment, from the repo root, after scripts/slm_results.py):
  .venv-train/Scripts/python -I scripts/make_manifest.py

Everything is read from files: the pins in src/pins.py, the data manifests, the training logs in
results/train_logs/, the run files in runs/ and the installed library versions. Nothing is typed in.
"""
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pins import BASE, BASE_REVISION, TUNED, TUNED_REVISION  # noqa: E402
import run_paper_bert as bert  # noqa: E402

DATA = ["data/sft/train_v1.jsonl", "data/sft/val.jsonl", "data/sft/train_v2.jsonl", "data/sft/train_v2_small.jsonl",
        "data/sft/train_v3.jsonl", "data/sft/teacher_v2.jsonl", "data/sft/teacher_v2_disputed.jsonl",
        "data/eval/E1000.jsonl", "data/eval/FULL.jsonl", "data/pressure/PRESSURE.jsonl"]
LIBRARIES = ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "huggingface-hub"]
FRONTIER = ["claude-opus-5-5__zero_shot_p1__low", "gpt-5.6-sol__few_shot__low"]  # each family's best cell


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def meta(cell: str) -> dict | None:
    path = ROOT / "runs" / f"{cell}.meta.json"
    return json.loads(path.read_text()) if path.exists() else None


def scored(run: str) -> dict:
    out = {}
    for eval_set in ("E1000", "PRESSURE"):
        m = meta(f"{run}__{eval_set}")
        if m:
            s, c = m["scores"], m["confusion"]
            out[eval_set] = {k: round(s[k], 4) for k in ("precision", "recall", "f1", "accuracy")} | {
                "correct": c["tp"] + c["tn"], "n": s["n"], "eval_sha256": m["eval_sha256"]}
    return out


def main() -> int:
    data_manifest = json.loads((ROOT / "data/manifest.json").read_text())
    sft_manifest = json.loads((ROOT / "data/sft/manifest.json").read_text())
    teacher = json.loads((ROOT / "data/sft/teacher_v2_meta.json").read_text())
    shipped_log = json.loads((ROOT / "results/train_logs/qwen3-1.7b-v2.json").read_text())
    args = shipped_log["args"]

    models = {"qwen3-1.7b-base": {"what": "untrained base", **scored("qwen3-1.7b-base__slm__na")}}
    for log in sorted((ROOT / "results/train_logs").glob("qwen3-*.json")):
        t = json.loads(log.read_text())
        models[log.stem] = {
            "train": {"file": t["train_file"], "sha256": t["train_sha256"], "rows": t["n"], "toxic": t["toxic"],
                      "seed": t["args"]["seed"], "steps": t["steps"], "loss": round(t["train_loss"], 4),
                      "minutes": t["minutes"], "code_commit": t["git_commit"]},
            **scored(f"{log.stem}__slm__na"),
        }
    models["qwen3-1.7b-v2-hub"] = {"what": f"{TUNED}@{TUNED_REVISION}, scored from the Hub", **scored("qwen3-1.7b-v2-hub__slm__na")}
    for cell in FRONTIER:
        models[cell] = {"what": "best prompted cell of its model family", **scored(cell)}
    models[bert.NAME] = {
        "what": "the L2DTnH paper's fine-tuned unitary/toxic-bert; it chose its epoch and threshold on the test split, so E1000 flatters it",
        "repo": bert.REPO, "revision": bert.COMMIT, "folder": bert.FOLDER, "sha256": bert.FILES,
        "threshold": bert.THRESHOLD, "max_length": bert.MAX_LEN, "train_rows": bert.TRAIN_ROWS, "eval_precision": "fp32 on the CPU",
        **scored(f"{bert.NAME}__classifier__na"),
    }

    manifest = {
        "project": "league-toxicity-slm",
        "behaviour": "One League of Legends chat line in; exactly one word out: toxic or nontoxic.",
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "student": {"base_model": BASE, "base_revision": BASE_REVISION, "method": "QLoRA (4-bit NF4 base, LoRA adapter)",
                    "lora": {"r": args["rank"], "alpha": args["alpha"], "dropout": 0.0, "targets": "all attention and MLP projections"},
                    "training": {k: args[k] for k in ("lr", "epochs", "batch", "grad_accum", "max_length")} | {"schedule": "cosine"},
                    "eval_precision": "bf16, greedy, batch 32"},
        "shipped": {"repo": TUNED, "revision": TUNED_REVISION, "trained_as": "qwen3-1.7b-v2",
                    "train_file": shipped_log["train_file"], "train_sha256": shipped_log["train_sha256"]},
        "teacher": {k: teacher[k] for k in ("teacher", "critic", "seed", "generated", "kept", "disputed", "cost_usd_list_price")},
        "seeds": {"train_val_split": sft_manifest["seed"], "teacher_generation": teacher["seed"],
                  "v2_mix": sft_manifest["train_v2"]["seed"], "v3_mix": sft_manifest["train_v3"]["seed"],
                  "training": sorted({m["train"]["seed"] for m in models.values() if "train" in m}), "bootstrap": 0},
        "source_data": {"repo": data_manifest["source"], "raw_sha256": data_manifest["raw"]},
        "data_sha256": {rel: sha256(rel) for rel in DATA},
        "versions": {"python": platform.python_version(), **{lib: metadata.version(lib) for lib in LIBRARIES}},
        "models": models,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"manifest.json: {len(models)} models, {len(DATA)} data files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
