"""Build the prompt-ceiling results table from the run files.

Usage: python -I src/report.py --eval-set E1000 [--include-limited]

Writes results/ceiling_<eval set>.md and, for the best model cell, results/errors_<cell>.md
listing every false alarm and miss. Every number comes from runs/*.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import bootstrap, confusion, scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BAR = {"precision": 0.80, "f1": 0.75}  # set in the README before any run
PAPER = [  # arXiv:2604.10175, GPT-4o (June 2025) on its 707-message subset (667 / 40)
    ("Paper: GPT-4o, P1 (toxic)", 0.3495, 0.9231, 0.5070),
    ("Paper: GPT-4o, P2 (inappropriate)", 0.3662, 0.6667, 0.4727),
]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cost_per_million(model: str, records: list[dict], prices: dict) -> str:
    if model not in prices or not records:
        return "n/a"
    p_in, p_out = prices[model]
    tokens_in = sum(r["input_tokens"] for r in records) / len(records)
    tokens_out = sum(r["output_tokens"] for r in records) / len(records)
    # Prices are per 1M tokens, so the average tokens per message times the price is the cost per 1M messages.
    return f"${(tokens_in * p_in + tokens_out * p_out):,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, choices=("E1000", "FULL"))
    ap.add_argument("--include-limited", action="store_true", help="also report smoke runs (__nN)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    prices = json.loads((ROOT / "src" / "prices.json").read_text())
    rows = []
    for meta_path in sorted((ROOT / "runs").glob(f"*__{args.eval_set}*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("limit") and not args.include_limited:
            continue
        records = load(ROOT / "runs" / f"{meta['cell']}.jsonl")
        c = confusion(records)
        s = scores(c)
        ci = bootstrap(records)
        latency = sorted(r["latency_ms"] for r in records)[len(records) // 2]
        rows.append({"meta": meta, "records": records, "c": c, "s": s, "ci": ci, "latency": latency})
    if not rows:
        print("no runs found")
        return 1
    rows.sort(key=lambda r: r["s"]["f1"], reverse=True)

    lines = [
        f"# Prompt-ceiling results: {args.eval_set}",
        "",
        f"Bar: precision ≥ {BAR['precision']:.2f} and F1 ≥ {BAR['f1']:.2f} (toxic = positive class). "
        "Failed replies (refused, unparseable, truncated, errored) count as wrong answers. "
        "Intervals are 95% bootstrap, 1,000 resamples.",
        "",
        "| Model | Prompt | Effort | Precision | Recall | F1 | Meets bar | Failures | Median latency | Cost per 1M messages |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        m, s, ci = r["meta"], r["s"], r["ci"]
        meets = s["precision"] >= BAR["precision"] and s["f1"] >= BAR["f1"]
        failures = ", ".join(f"{k} {v}" for k, v in r["c"]["failures"].items()) or "0"
        lines.append(
            f"| {m['model']} | {m['strategy']} | {m['effort']} "
            f"| {s['precision']:.3f} ({ci['precision'][0]:.2f}–{ci['precision'][1]:.2f}) "
            f"| {s['recall']:.3f} ({ci['recall'][0]:.2f}–{ci['recall'][1]:.2f}) "
            f"| {s['f1']:.3f} ({ci['f1'][0]:.2f}–{ci['f1'][1]:.2f}) "
            f"| {'yes' if meets else 'no'} | {failures} | {r['latency']} ms "
            f"| {cost_per_million(m['model'], r['records'], prices)} |"
        )
    lines += ["", "Paper rows for reference (measured on a different 707-message subset, so the comparison is approximate):", ""]
    lines += ["| Source | Precision | Recall | F1 |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {p:.3f} | {rc:.3f} | {f:.3f} |" for name, p, rc, f in PAPER]

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"ceiling_{args.eval_set}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    best = next((r for r in rows if not r["meta"]["model"].startswith("baseline:")), None)
    if best:
        fps = [x for x in best["records"] if x["gold"] == 0 and x["pred"] != 0]
        fns = [x for x in best["records"] if x["gold"] == 1 and x["pred"] != 1]
        err = [f"# Errors: {best['meta']['cell']}", "", f"## False alarms ({len(fps)})", ""]
        err += [f"- `{x['message']}` ({x['status']})" for x in fps]
        err += ["", f"## Misses ({len(fns)})", ""]
        err += [f"- `{x['message']}` ({x['status']})" for x in fns]
        (out_dir / f"errors_{best['meta']['cell']}.md").write_text("\n".join(err) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
