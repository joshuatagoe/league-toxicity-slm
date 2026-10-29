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
        if meta.get("limit"):
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
        "Robustness is accuracy on the 40-message pressure set (labelled by the project author; 10 toxic, 30 not). "
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
    (ROOT / "results" / "ablation1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
