"""Base vs tuned on one eval set, in one command: the comparison a grader runs on a held-out file.

Usage (training environment, from the repo root):
  .venv-train/Scripts/python compare.py --eval-set path/to/heldout.csv

Runs eval.py twice on the same file, the untrained Qwen3-1.7B and the public v2 adapter, both at
the commits pinned in src/pins.py. Each run writes its per-line answers to runs/. Then prints both
models' scores on the labelled rows, the paired difference with a 95% interval, and the lines where
the two models disagree. The file can be JSON Lines, JSON, CSV, TSV or plain text (src/evalsets.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import confusion, paired, scores  # noqa: E402
from pins import BASE, BASE_REVISION, TUNED, TUNED_REVISION  # noqa: E402

WORD = {1: "toxic", 0: "nontoxic", None: "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, help="the file to compare on")
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--show", type=int, default=40, help="disagreements to print")
    args = ap.parse_args()

    eval_path = Path(args.eval_set)
    models = [("qwen3-1.7b-base", BASE, BASE_REVISION), ("qwen3-1.7b-v2-hub", TUNED, TUNED_REVISION)]
    runs = {}
    for name, model, revision in models:
        cmd = [sys.executable, "-u", str(ROOT / "eval.py"), "--model", model, "--revision", revision,
               "--name", name, "--eval-set", str(eval_path), "--batch", str(args.batch)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        cell = f"{name}__slm__na__{eval_path.stem}" + (f"__n{args.limit}" if args.limit else "")
        runs[name] = [json.loads(l) for l in (ROOT / "runs" / f"{cell}.jsonl").open(encoding="utf-8") if l.strip()]

    base, tuned = runs["qwen3-1.7b-base"], runs["qwen3-1.7b-v2-hub"]
    labelled = [i for i, r in enumerate(base) if r["gold"] is not None]
    print(f"\n{eval_path.name}: {len(base)} lines, {len(labelled)} with a label")
    print(f"{'model':<22}{'precision':>10}{'recall':>8}{'F1':>8}{'accuracy':>10}{'said toxic':>12}")
    for label, recs in (("untrained", base), ("tuned (v2)", tuned)):
        said = sum(r["pred"] == 1 for r in recs)
        if labelled:
            s = scores(confusion([recs[i] for i in labelled]))
            print(f"{label:<22}{s['precision']:>10.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['accuracy']:>10.3f}{said:>12}")
        else:
            print(f"{label:<22}{'':>36}{said:>12}")
    if labelled:
        print("\ntuned minus untrained, same lines resampled (95% interval):")
        for metric in ("precision", "f1", "accuracy"):
            point, lo, hi = paired([tuned[i] for i in labelled], [base[i] for i in labelled], metric)
            print(f"  {metric:<10}{point:+.3f}  ({lo:+.3f} to {hi:+.3f})")

    differ = [(b, t) for b, t in zip(base, tuned) if b["pred"] != t["pred"]]
    print(f"\n{len(differ)} lines where the two disagree" + (f" (first {args.show})" if len(differ) > args.show else "") + ":")
    print(f"  {'label':<10}{'untrained':<11}{'tuned':<10}message")
    for b, t in differ[: args.show]:
        print(f"  {WORD[b['gold']]:<10}{WORD[b['pred']]:<11}{WORD[t['pred']]:<10}{b['message'][:70]}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
