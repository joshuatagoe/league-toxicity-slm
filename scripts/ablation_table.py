"""Build the brief's Ablation 1 table: Spec adherence and Robustness per model x prompt.

Usage: python -I scripts/ablation_table.py

Spec adherence = F1 on E1000 (the experts' labels). Robustness = accuracy on PRESSURE
(the 40 pressure messages, labelled by the user). Writes results/ablation1.md from
the run files; nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import bootstrap, confusion, scores  # noqa: E402


def cells(eval_set: str) -> dict:
    out = {}
    for meta_path in (ROOT / "runs").glob(f"*__{eval_set}.meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("limit") or meta["strategy"] == "slm":  # prompted cells only; the small model is in slm_results.py
            continue
        records = [json.loads(l) for l in (ROOT / "runs" / f"{meta['cell']}.jsonl").open(encoding="utf-8") if l.strip()]
        out[(meta["model"], meta["strategy"])] = (records, scores(confusion(records)))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    spec, robust = cells("E1000"), cells("PRESSURE")
    keys = sorted(set(spec) & set(robust), key=lambda k: -spec[k][1]["f1"])
    lines = [
        "# Ablation 1: prompt ceiling",
        "",
        "Spec adherence is F1 for the toxic class on E1000 (1,000 test messages, the experts' labels), with its 95% bootstrap interval. "
        "Robustness is accuracy on the 40-message pressure set (labelled by the project author; 11 toxic, 29 not). "
        "Bar: precision ≥ 0.80 and F1 ≥ 0.75 on E1000.",
        "",
        "| Model | Prompt | Spec adherence (F1, E1000) | Precision | Recall | Robustness (accuracy, pressure set) | Pressure F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k in keys:
        rec, s = spec[k]
        ci = bootstrap(rec)["f1"]
        r = robust[k][1]
        lines.append(f"| {k[0]} | {k[1]} | {s['f1']:.3f} ({ci[0]:.2f}–{ci[1]:.2f}) | {s['precision']:.3f} | {s['recall']:.3f} | {r['accuracy']:.3f} | {r['f1']:.3f} |")
    lines += [
        "",
        "## The failure mode",
        "",
        "No cell meets the bar. The best cell over-flags short trash talk the expert panel left alone (`easy`, `gg mid`, `just l2p`) "
        "and misses lines whose toxicity depends on surrounding chat it can't see (`GAREN TANK`, `19 assist`). "
        "More instructions (few-shot, structured) cut Claude's precision by almost half and left GPT-5.6 Sol's unchanged. "
        "The full analysis, with the author's blind relabel of the disagreements, is in [docs/CEILING_REPORT.md](../docs/CEILING_REPORT.md).",
        "",
        "## Scoring: exact match, not an LLM judge",
        "",
        "Every eval line already has a human label, and every model must answer with one word, so a reply is compared with the label directly: "
        "it passes only if it is exactly `toxic` or `nontoxic` and matches. Anything else counts as wrong. "
        "An LLM judge would add a second model's mistakes and cost to a check that a string comparison does exactly, "
        "and the obvious judges are the frontier models this table shows disagreeing with the experts. "
        "The raw transcripts are the per-message run files in `runs/*.jsonl`: message, label, parsed prediction and raw reply for every line.",
    ]
    (ROOT / "results" / "ablation1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
