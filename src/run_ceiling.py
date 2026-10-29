"""Run one model x prompt-style cell of the prompt-ceiling grid.

Usage:
  python -I src/run_ceiling.py --model claude-opus-5-5 --strategy structured --eval-set E707
  python -I src/run_ceiling.py --model baseline:keyword --strategy zero_shot_p1 --eval-set E707

Writes runs/<cell>.jsonl (one line per message) and runs/<cell>.meta.json. Re-running the
same command resumes: messages already in the .jsonl are skipped.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script's own folder
from labels import parse_label  # noqa: E402
from metrics import confusion, scores  # noqa: E402
from providers import classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ("zero_shot_p1", "few_shot", "structured")


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env (git-ignored) without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_template(strategy: str) -> str:
    template = (ROOT / "prompts" / f"{strategy}.txt").read_text(encoding="utf-8")
    if strategy == "few_shot":
        examples = json.loads((ROOT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))
        block = "\n\n".join(f"Message: {e['message']}\nLabel: {e['label']}" for e in examples)
        template = template.replace("{examples}", block)
    return template


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--eval-set", required=True, choices=("E707", "FULL"))
    ap.add_argument("--limit", type=int, help="score only the first N messages (smoke runs)")
    ap.add_argument("--effort", default="low", help="reasoning effort for models that take one")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    rows = load_jsonl(ROOT / "data" / "eval" / f"{args.eval_set}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    template = build_template(args.strategy)
    prompt_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
    effort = "na" if args.model.startswith("baseline:") else args.effort

    cell = f"{args.model.replace(':', '-')}__{args.strategy}__{effort}__{args.eval_set}"
    if args.limit:
        cell += f"__n{args.limit}"
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    out_path = runs / f"{cell}.jsonl"
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    pending = [r for r in rows if r["id"] not in done]
    print(f"{cell}: {len(rows)} messages, {len(done)} already scored, {len(pending)} to run")

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    meta = {
        "cell": cell,
        "model": args.model,
        "strategy": args.strategy,
        "effort": effort,
        "eval_set": args.eval_set,
        "limit": args.limit,
        "eval_sha256": manifest["eval"][f"{args.eval_set}.jsonl"],
        "prompt_sha256_12": prompt_sha,
        "git_commit": git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    lock = threading.Lock()

    def score(row: dict) -> dict:
        prompt = template.replace("{text}", row["message"])
        reply = classify(args.model, prompt, row["message"], args.effort)
        pred = parse_label(reply.text, structured=args.strategy == "structured") if reply.status == "ok" else None
        status = reply.status if reply.status != "ok" else ("ok" if pred is not None else "parse_fail")
        return {
            "id": row["id"],
            "message": row["message"],
            "gold": row["label"],
            "pred": pred,
            "status": status,
            "raw": reply.text,
            "stop_reason": reply.stop_reason,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "latency_ms": reply.latency_ms,
            "error": reply.error,
        }

    with out_path.open("a", encoding="utf-8", newline="\n") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(score, row) for row in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            record = fut.result()
            with lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(pending)}")

    records = load_jsonl(out_path)
    by_id = {r["id"] for r in rows}
    records = [r for r in records if r["id"] in by_id]
    c = confusion(records)
    s = scores(c)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["confusion"] = c
    meta["scores"] = s
    (runs / f"{cell}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"precision {s['precision']:.3f}  recall {s['recall']:.3f}  f1 {s['f1']:.3f}  accuracy {s['accuracy']:.3f}  n {s['n']}  failures {c['failures']}")
    if len(records) != len(rows):
        print(f"WARNING: {len(records)} scored rows for {len(rows)} messages")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
