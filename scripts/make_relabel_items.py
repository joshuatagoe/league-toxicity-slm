"""Build the relabelling set and inject it into relabel/index.html.

Usage: python -I scripts/make_relabel_items.py

Part A (relabel): every disagreement between the experts and the best frontier cell
(Claude Opus 5.5, zero-shot, on E1000), plus agreement controls (15 both-toxic,
14 both-non-toxic), shuffled. Each item carries up to 4 chat lines before and 2 after
from the same chat log, as the original annotators could see. Neither the experts'
label nor the model's answer goes into the page.

Part B (pressure): data/pressure/candidates.json, hand-written pressure messages
for the Robustness column, labelled by the user.

Writes relabel/key.json (the answer key, never shown in the page) and rewrites the
ITEMS block in relabel/index.html.
"""
import csv
import json
import random
import re
from pathlib import Path

SEED = 29
CONTROLS = {"tp": 15, "tn": 14}
BEFORE, AFTER = 4, 2
BEST_CELL = "claude-opus-5-5__zero_shot_p1__low__E1000"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def read_csv(name: str) -> list[dict]:
    with (RAW / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = read_csv("2_16000_chatlogs_english_only_train_split.csv") + read_csv("2_16000_chatlogs_english_only_test_split.csv")
logs: dict[str, list[dict]] = {}
for r in rows:
    logs.setdefault(r["chatlog_id"], []).append(r)
for lines in logs.values():
    lines.sort(key=lambda r: (r["time"], int(float(r["id"]))))
by_id = {r["id"]: r for r in rows}


def line(r: dict) -> dict:
    return {"time": r["time"], "side": r["association_to_offender"] or "unknown", "champion": r["champion_name"], "text": r["message"]}


def context(message_id: str) -> dict:
    target = by_id[message_id]
    lines = logs[target["chatlog_id"]]
    i = next(k for k, r in enumerate(lines) if r["id"] == message_id)
    return {
        "before": [line(r) for r in lines[max(0, i - BEFORE) : i]],
        "target": line(target),
        "after": [line(r) for r in lines[i + 1 : i + 1 + AFTER]],
    }


with (ROOT / "runs" / f"{BEST_CELL}.jsonl").open(encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

disagree = [r for r in records if r["pred"] != r["gold"]]
rng = random.Random(SEED)
tp = rng.sample([r for r in records if r["gold"] == 1 and r["pred"] == 1], CONTROLS["tp"])
tn = rng.sample([r for r in records if r["gold"] == 0 and r["pred"] == 0], CONTROLS["tn"])

key, items = {}, []
for group, recs in (("disagreement", disagree), ("control", tp + tn)):
    for r in recs:
        item_id = f"r-{r['id']}"
        items.append({"id": item_id, "part": "relabel", **context(r["id"])})
        key[item_id] = {"group": group, "expert": r["gold"], "model": r["pred"], "message": r["message"]}
rng.shuffle(items)

pressure = json.loads((ROOT / "data" / "pressure" / "candidates.json").read_text(encoding="utf-8"))
for p in pressure:
    items.append({"id": p["id"], "part": "pressure", "target": {"text": p["message"]}})
    key[p["id"]] = {"group": "pressure", "kind": p["kind"], "message": p["message"]}

out_dir = ROOT / "relabel"
(out_dir / "key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

page = out_dir / "index.html"
html = page.read_text(encoding="utf-8")
payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
html, n = re.subn(r"(/\*ITEMS\*/).*?(/\*END ITEMS\*/)", lambda m: m.group(1) + payload + m.group(2), html, flags=re.S)
if n != 1:
    raise SystemExit("ITEMS block not found in relabel/index.html")
page.write_text(html, encoding="utf-8", newline="\n")
print(f"{len(disagree)} disagreements + {len(tp) + len(tn)} controls + {len(pressure)} pressure items -> relabel/index.html")
