"""Score the user's blind relabels against the experts and Claude, and freeze the pressure set.

Usage: python -I scripts/score_relabel.py

Reads relabel/responses/labels/*.json (exported from the relabelling page) and
relabel/key.json. Writes results/relabel_analysis.md and data/pressure/PRESSURE.jsonl
(the pressure messages the user labelled toxic or not toxic; Unsure ones are left out).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = {"toxic": 1, "nontoxic": 0}


def kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    key = json.loads((ROOT / "relabel" / "key.json").read_text(encoding="utf-8"))
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "relabel" / "responses" / "labels").glob("*.json")}
    missing = sorted(set(key) - set(labels))
    if missing:
        print(f"{len(missing)} items have no label yet: {missing[:5]}")
        return 1

    you = {k: labels[k]["label"] for k in key}
    relabel = {k: v for k, v in key.items() if v["group"] != "pressure"}
    lines = ["# Relabel check: your labels vs the experts and Claude", ""]
    lines.append("Blind relabel of 100 messages from E1000: Claude Opus 5.5's 71 disagreements with the experts plus 29 agreement controls. "
                 "\"Claude\" is the best frontier cell (zero-shot). Unsure labels are left out of the agreement rates.")
    lines += ["", "## Your labels", ""]
    dist = Counter(you[k] for k in relabel)
    lines.append(f"Toxic {dist['toxic']}, not toxic {dist['nontoxic']}, unsure {dist['unsure']} (of {len(relabel)}).")

    def sided(items: dict) -> dict:
        out = Counter()
        for k, v in items.items():
            if you[k] == "unsure":
                out["unsure"] += 1
            elif CODE[you[k]] == v["expert"]:
                out["experts"] += 1
            else:
                out["claude"] += 1
        return out

    groups = {
        "Claude's false alarms (experts: fine, Claude: toxic)": {k: v for k, v in relabel.items() if v["group"] == "disagreement" and v["expert"] == 0},
        "Claude's misses (experts: toxic, Claude: fine)": {k: v for k, v in relabel.items() if v["group"] == "disagreement" and v["expert"] == 1},
        "Controls where both said toxic": {k: v for k, v in relabel.items() if v["group"] == "control" and v["expert"] == 1},
        "Controls where both said fine": {k: v for k, v in relabel.items() if v["group"] == "control" and v["expert"] == 0},
    }
    lines += ["", "## Whose side you took", "", "| Group | Messages | Sided with the experts | Sided with Claude | Unsure |", "| --- | --- | --- | --- | --- |"]
    for name, items in groups.items():
        s = sided(items)
        if "Controls" in name:
            lines.append(f"| {name} | {len(items)} | {s['experts']} (agreed with both) | {s['claude']} (disagreed with both) | {s['unsure']} |")
        else:
            lines.append(f"| {name} | {len(items)} | {s['experts']} | {s['claude']} | {s['unsure']} |")

    decided = [k for k in relabel if you[k] != "unsure"]
    exp_pairs = [(CODE[you[k]], relabel[k]["expert"]) for k in decided]
    mod_pairs = [(CODE[you[k]], relabel[k]["model"]) for k in decided]
    agree_e = sum(a == b for a, b in exp_pairs) / len(exp_pairs)
    agree_m = sum(a == b for a, b in mod_pairs) / len(mod_pairs)
    lines += ["", "## Agreement on the decided messages", "",
              f"- With the experts: {agree_e:.0%} (Cohen's kappa {kappa(exp_pairs):.2f}, n {len(exp_pairs)})",
              f"- With Claude: {agree_m:.0%} (Cohen's kappa {kappa(mod_pairs):.2f}, n {len(mod_pairs)})",
              "", "These 100 messages were chosen for disagreement, so these rates describe the hard cases, not the dataset."]

    lines += ["", "## Every disagreement and your label", "", "| Message | Experts | Claude | You | Note |", "| --- | --- | --- | --- | --- |"]
    word = {1: "toxic", 0: "fine"}
    for k, v in sorted(relabel.items(), key=lambda kv: (kv[1]["group"], -kv[1]["expert"], kv[1]["message"].lower())):
        if v["group"] != "disagreement":
            continue
        note = labels[k].get("note", "").replace("|", "/")
        lines.append(f"| `{v['message'].replace('|', '/')}` | {word[v['expert']]} | {word[v['model']]} | {you[k]} | {note} |")

    pressure = {k: v for k, v in key.items() if v["group"] == "pressure"}
    out = ROOT / "data" / "pressure" / "PRESSURE.jsonl"
    kept = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for k, v in sorted(pressure.items()):
            if you[k] == "unsure":
                continue
            f.write(json.dumps({"id": k, "message": v["message"], "label": CODE[you[k]], "kind": v["kind"]}, ensure_ascii=False) + "\n")
            kept += 1
    lines += ["", "## Pressure set", "", f"{kept} of {len(pressure)} pressure messages have a toxic or not-toxic label and form `data/pressure/PRESSURE.jsonl`.", "",
              "| Kind | Message | Your label |", "| --- | --- | --- |"]
    for k, v in sorted(pressure.items()):
        lines.append(f"| {v['kind']} | `{v['message']}` | {you[k]} |")

    report = ROOT / "results" / "relabel_analysis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:24]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
