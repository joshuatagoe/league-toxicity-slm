"""Why does the prompted frontier model catch sarcasm that the tuned small model misses?

Two tests on the same lines, written up in results/sarcasm_probe.md:
1. Confidence. The probability the small model gives `toxic` as its first answer token, under the
   eval prompt, for the untrained base, v1, v2 and v3. If sarcastic lines get clearly higher odds
   than sincere praise but stay under one half, the model sees the sarcasm and its training labels
   outvote it. The same numbers give a threshold sweep on E1000 and the pressure set.
2. A different question. "Is this line sarcastic? yes or no", asked of the same models and of
   Claude Opus 5.5. If the untrained model can tell sarcastic lines from sincere ones, its toxic
   misses come from the standard it was taught, not from what a 1.7B model can see.

The lines: every test-split message with a praise word (gj, ty, nice, ...), which no model trained
on, once each, plus the pressure set's four sarcasm lines. Which test-split lines read as sarcastic
was marked by Claude, the coding assistant, in the session that wrote this script (SARCASTIC and
UNCLEAR below), so the marks are one reader's call.

Usage (from the repo root):
  .venv-train/Scripts/python -I src/run_sarcasm_probe.py qwen     # GPU: both tests for the Qwen models
  .venv/Scripts/python -I src/run_sarcasm_probe.py claude         # test 2 for Claude; ANTHROPIC_API_KEY in .env, about $0.20
  python -I src/run_sarcasm_probe.py report                       # results/sarcasm_probe.md
"""
import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script's own folder
from metrics import confusion, scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sarcasm"
PRAISE = re.compile(r"\b(gj|nice|great|good job|wp|well played|thanks|thx|ty|genius|amazing|incredible|gz|grats|love|"
                    r"awesome|perfect|brilliant|smart|clutch|helpful|useful|congrats|bravo)\b", re.I)
# Test-split ids, marked by reading each of the 190 praise-word lines.
SARCASTIC = {"11035": "feed him more ty seju", "395": "so smart.... gl hf", "4195": "ty for lose", "7637": "ty for no help",
             "3631": "nice u took ignite to ks tnx", "6050": "thanks for fking my game dude", "11963": "nice feed ty",
             "9435": "nice so no ward bot?", "5453": "thx for SS --'", "8949": "who lost lane phase by smart move?"}
UNCLEAR = {"15286": "Ty for Q farm", "724": "nice attackspeed vayne xD", "9020": "ty for cayy me", "5890": "really great team",
           "2611": "thx for your great help", "8228": "thx for help garen..."}
# Praise next to a failure, for counting how the experts labelled that shape in the training data.
FAILURE = re.compile(r"(\bfeed|\blos[et]|no help|\bks\b|steal|stole|\bafk|\btroll|\b0/|\bint\b|throw|died|\bdeath|ignite|"
                     r"flash|no ward|\bfail|\bmiss|\bdive)", re.I)
MODELS = {"untrained": None, "v1": "models/qwen3-1.7b-v1-n10919", "v2": "models/qwen3-1.7b-v2", "v3": "models/qwen3-1.7b-v3"}
SARCASM_SYSTEM = ("You read League of Legends chat. Is the player's message sarcastic: does it say the opposite of what it "
                  "means, such as mock praise? Reply with exactly one word: yes or no.")
CLAUDE = "claude-opus-5-5"
CLAUDE_PROMPT = ("Is this League of Legends chat message sarcastic: does it say the opposite of what it means, such as mock "
                 "praise? Reply with exactly one word: yes or no.\n\nMessage: {message}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def probe_lines() -> list[dict]:
    """The pressure set's sarcasm lines, then each distinct praise-word message of the test split."""
    lines = []
    for r in read_jsonl(ROOT / "data" / "pressure" / "PRESSURE.jsonl"):
        if r["kind"] == "sarcasm at a teammate":  # the one labelled fine "depends on unseen context" in the rules
            lines.append({"id": r["id"], "source": "pressure", "message": r["message"], "label": r["label"],
                          "labelled_by": "author", "sarcastic": "yes" if r["label"] else "unclear"})
    seen = set()
    for r in read_jsonl(ROOT / "data" / "eval" / "FULL.jsonl"):
        m = r["message"] or ""
        if PRAISE.search(m) and m not in seen:
            seen.add(m)
            mark = "yes" if r["id"] in SARCASTIC else "unclear" if r["id"] in UNCLEAR else "no"
            lines.append({"id": r["id"], "source": "test split", "message": m, "label": r["label"],
                          "labelled_by": "experts", "sarcastic": mark})
    marked = {l["id"] for l in lines if l["source"] == "test split" and l["sarcastic"] != "no"}
    assert marked == set(SARCASTIC) | set(UNCLEAR), marked ^ (set(SARCASTIC) | set(UNCLEAR))
    return lines


def run_qwen() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pins import BASE, BASE_REVISION
    from slm import render

    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16, device_map={"": 0})
    adapters = [(name, path) for name, path in MODELS.items() if path]
    model = PeftModel.from_pretrained(model, str(ROOT / adapters[0][1]), adapter_name=adapters[0][0])
    for name, path in adapters[1:]:
        model.load_adapter(str(ROOT / path), adapter_name=name)
    model.eval()

    def first_tokens(words: list[str]) -> list[int]:
        return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})

    toxic, nontoxic = first_tokens(["toxic", "Toxic"]), first_tokens(["nontoxic", "Nontoxic", "non-toxic"])
    yes, no = first_tokens(["yes", "Yes"]), first_tokens(["no", "No"])
    assert not set(toxic) & set(nontoxic) and not set(yes) & set(no)

    def sarcasm_prompt(text: str) -> str:
        msgs = [{"role": "system", "content": SARCASM_SYSTEM}, {"role": "user", "content": text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def odds(prompts: list[str], pos: list[int], neg: list[int], batch: int = 32) -> list[list[float]]:
        """[P(pos) / (P(pos) + P(neg)), P(pos) + P(neg)] for the first answer token of each prompt."""
        out = []
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # as generate() does with left padding
            probs = model(**enc, position_ids=position_ids).logits[:, -1, :].float().softmax(-1)
            p, n = probs[:, pos].sum(-1), probs[:, neg].sum(-1)
            out += [[round(float(a / (a + b)), 6), round(float(a + b), 6)] for a, b in zip(p, n)]
        return out

    probe = probe_lines()
    sets = {"E1000": read_jsonl(ROOT / "data" / "eval" / "E1000.jsonl"), "PRESSURE": read_jsonl(ROOT / "data" / "pressure" / "PRESSURE.jsonl"),
            "probe": probe}
    result = {"base": BASE, "base_revision": BASE_REVISION, "adapters": MODELS, "toxic": {}, "sarcastic": {}}
    for name in MODELS:
        if name != "untrained":
            model.set_adapter(name)
        with model.disable_adapter() if name == "untrained" else nullcontext():
            result["toxic"][name] = {s: dict(zip([r["id"] for r in rows], odds([render(tok, r["message"] or "") for r in rows], toxic, nontoxic)))
                                     for s, rows in sets.items()}
            result["sarcastic"][name] = dict(zip([r["id"] for r in probe], odds([sarcasm_prompt(r["message"]) for r in probe], yes, no)))
        print(name, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen.json").write_text(json.dumps(result) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'qwen.json'}")
    return 0


def run_claude() -> int:
    from providers import _call_anthropic
    from run_ceiling import load_dotenv

    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "claude.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for line in probe_lines():
            if line["id"] in done:
                continue
            reply = _call_anthropic(CLAUDE, CLAUDE_PROMPT.format(message=line["message"]), "low")
            word = re.sub(r"[^a-z]", "", reply.text.strip().split()[0].lower()) if reply.text.strip() else ""
            f.write(json.dumps({"id": line["id"], "answer": word if word in ("yes", "no") else None, "raw": reply.text,
                                "status": reply.status, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}) + "\n")
            f.flush()
    rows = read_jsonl(path)
    cost = sum(r["input_tokens"] * 4.0 + r["output_tokens"] * 20.0 for r in rows) / 1e6  # src/prices.json, USD per million
    print(f"{path}: {len(rows)} answers, {sum(r['answer'] is None for r in rows)} unreadable, ${cost:.3f} at list prices")
    return 0


def auc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random line from pos scores above a random line from neg (ties count half)."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    probe = probe_lines()
    q = json.loads((OUT / "qwen.json").read_text(encoding="utf-8"))
    claude = {r["id"]: r["answer"] for r in read_jsonl(OUT / "claude.jsonl")}
    claude_toxic = {r["id"]: r["pred"] for c in ("FULL", "PRESSURE")
                    for r in read_jsonl(ROOT / "runs" / f"{CLAUDE}__zero_shot_p1__low__{c}.jsonl")}
    group = {g: [l for l in probe if l["sarcastic"] == g] for g in ("yes", "unclear", "no")}
    n = {g: len(v) for g, v in group.items()}
    tox = lambda m, l: q["toxic"][m]["probe"][l["id"]][0]  # noqa: E731
    above = lambda p, t: round(p, 3) > t  # bf16 logits can't separate odds closer than this  # noqa: E731
    sar = lambda m, l: q["sarcastic"][m][l["id"]][0]  # noqa: E731

    train = read_jsonl(ROOT / "data" / "sft" / "train_v1.jsonl")
    shaped = [(r["label"], r["prompt"][-1]["content"]) for r in train if PRAISE.search(r["prompt"][-1]["content"]) and FAILURE.search(r["prompt"][-1]["content"])]
    lines = [
        "# Sarcasm probe: does the small model see sarcasm, or was it taught not to flag it?",
        "",
        "Generated by `src/run_sarcasm_probe.py` from `results/sarcasm/`. The lines: the pressure set's 4 sarcasm lines (written and "
        "labelled by the author) and every distinct test-split message with a praise word such as `gj`, `ty` or `nice` (labelled by the "
        "L2DTnH experts; no model trained on the test split). "
        f"Of the {n['yes'] + n['unclear'] + n['no']} lines, {n['yes']} read as sarcastic, {n['unclear']} could be either and {n['no']} read as sincere "
        "or neutral. The test-split marks are one reader's call: Claude, the coding assistant, in the session that wrote the script.",
        "",
        "## How the labels treat sarcasm",
        "",
        f"- **Test split:** {sum(l['label'] for l in group['yes'] if l['source'] == 'test split')} of the "
        f"{sum(l['source'] == 'test split' for l in group['yes'])} sarcastic lines are labelled toxic by the experts: "
        + ", ".join(f"`{l['message']}`" for l in group["yes"] if l["source"] == "test split" and l["label"]) + ". Labelled fine: "
        + ", ".join(f"`{l['message']}`" for l in group["yes"] if l["source"] == "test split" and not l["label"]) + ".",
        f"- **Training data (`train_v1.jsonl`):** {len(shaped)} lines put praise next to a failure word (feed, lose, ks, afk, no help, ...). "
        f"The experts labelled {sum(l for l, _ in shaped)} toxic. Similar lines get both labels: "
        + ", ".join(f"`{m}` ({'toxic' if l else 'fine'})" for l, m in shaped if m.lower() in ("nice feed cait", "gj feed cait", "nice feed mf bot", "thanks for no help at all", "ty fro feed", "nice feed ty")) + ".",
        "- **Pressure set:** the written rules call sarcasm that mocks a teammate toxic, so its 3 sarcastic lines are labelled toxic.",
        "",
        "## Test 2: asked whether a line is sarcastic",
        "",
        "Each model gets a different question, \"Is this line sarcastic? yes or no\". For the Qwen models, a line counts as called "
        "sarcastic when `yes` is more likely than `no` as the first answer token; AUC is the chance that a random sarcastic line gets higher "
        "odds than a random sincere one (0.5 is chance, 1.0 perfect). Claude answers in words, so it has no AUC.",
        "",
        f"| Model | Sarcastic lines called sarcastic (of {n['yes']}) | Unclear (of {n['unclear']}) | Sincere lines called sarcastic (of {n['no']}) | AUC, sarcastic vs sincere |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in q["sarcastic"]:
        said = {g: sum(above(sar(m, l), 0.5) for l in v) for g, v in group.items()}
        lines.append(f"| Qwen3-1.7B {m} | {said['yes']} | {said['unclear']} | {said['no']} | "
                     f"{auc([sar(m, l) for l in group['yes']], [sar(m, l) for l in group['no']]):.2f} |")
    said = {g: sum(claude.get(l["id"]) == "yes" for l in v) for g, v in group.items()}
    lines.append(f"| Claude Opus 5.5 | {said['yes']} | {said['unclear']} | {said['no']} | — |")
    catch_all = {m: sum(sar(m, l) >= min(sar(m, s) for s in group["yes"]) for l in group["no"]) for m in q["sarcastic"]}
    lines += ["", "Claude called every sarcastic line sarcastic, and many sincere ones too. For a like-for-like comparison: with the bar "
              f"for `yes` set low enough to catch all {n['yes']} sarcastic lines, each Qwen model also calls this many of the {n['no']} sincere "
              "lines sarcastic: " + ", ".join(f"{m} {c}" for m, c in catch_all.items()) + f"; Claude {said['no']}."]

    lines += [
        "",
        "## Test 1: how toxic each model thinks the lines are",
        "",
        "The probability of `toxic` as the first answer token under the eval prompt, so the same question the models were scored on. "
        "A line is flagged when it is above 0.5. For scale, the last two columns give the medians over `E1000`'s expert-labelled toxic and "
        "fine lines.",
        "",
        "| Model | Median, sarcastic | Median, sincere | AUC, sarcastic vs sincere | Sarcastic flagged | Sincere flagged | Median, E1000 toxic | Median, E1000 fine |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    e1000 = {r["id"]: r["label"] for r in read_jsonl(ROOT / "data" / "eval" / "E1000.jsonl")}
    for m in q["toxic"]:
        ys, ns = [tox(m, l) for l in group["yes"]], [tox(m, l) for l in group["no"]]
        et = [q["toxic"][m]["E1000"][i][0] for i, y in e1000.items() if y]
        ef = [q["toxic"][m]["E1000"][i][0] for i, y in e1000.items() if not y]
        lines.append(f"| Qwen3-1.7B {m} | {median(ys):.3f} | {median(ns):.3f} | {auc(ys, ns):.2f} | {sum(above(p, 0.5) for p in ys)} of {len(ys)} | "
                     f"{sum(above(p, 0.5) for p in ns)} of {len(ns)} | {median(et):.3f} | {median(ef):.3f} |")
    lines.append(f"| Claude Opus 5.5 (its eval answers) | — | — | — | {sum(claude_toxic.get(l['id']) == 1 for l in group['yes'])} of {n['yes']} | "
                 f"{sum(claude_toxic.get(l['id']) == 1 for l in group['no'])} of {n['no']} | — | — |")

    pressure = {r["id"]: r for r in read_jsonl(ROOT / "data" / "pressure" / "PRESSURE.jsonl")}
    lines += [
        "",
        "## Lowering the bar: v2 and v3 at other thresholds",
        "",
        "If the tuned models see sarcasm but keep it under 0.5, a lower threshold would catch it. What that costs on the benchmark:",
        "",
        f"| Model | Threshold | E1000 precision | E1000 recall | E1000 F1 | Pressure set (of 40) | Sarcastic lines flagged (of {n['yes']}) | Sincere lines flagged (of {n['no']}) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in ("v2", "v3"):
        for t in (0.5, 0.3, 0.2, 0.1, 0.05):
            s = scores(confusion([{"gold": e1000[i], "pred": int(above(p, t))} for i, (p, _) in q["toxic"][m]["E1000"].items()]))
            right = sum(int(above(p, t)) == pressure[i]["label"] for i, (p, _) in q["toxic"][m]["PRESSURE"].items())
            lines.append(f"| {m} | {t} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {right} | "
                         f"{sum(above(tox(m, l), t) for l in group['yes'])} | {sum(above(tox(m, l), t) for l in group['no'])} |")

    agree = {}
    for m, cell in (("untrained", "qwen3-1.7b-base"), ("v1", "qwen3-1.7b-v1-n10919"), ("v2", "qwen3-1.7b-v2"), ("v3", "qwen3-1.7b-v3")):
        preds = {r["id"]: r["pred"] for r in read_jsonl(ROOT / "runs" / f"{cell}__slm__na__E1000.jsonl")}
        agree[m] = sum(int(above(p, 0.5)) == preds[i] for i, (p, _) in q["toxic"][m]["E1000"].items())
    word = {1: "toxic", 0: "fine"}
    mass ={m: min(v[1] for s in q["toxic"][m].values() for v in s.values()) for m in q["toxic"]}
    lines += [
        "",
        "## Every sarcastic and unclear line",
        "",
        "| Line | Source | Label | P(toxic) untrained / v2 / v3 | P(sarcastic) untrained / v2 | Claude: toxic? | Claude: sarcastic? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in ("yes", "unclear"):
        for l in group[g]:
            lines.append(f"| `{l['message']}`{' (unclear)' if g == 'unclear' else ''} | {l['source']} | {'toxic' if l['label'] else 'fine'} ({l['labelled_by']}) | "
                         f"{tox('untrained', l):.2f} / {tox('v2', l):.2f} / {tox('v3', l):.2f} | {sar('untrained', l):.2f} / {sar('v2', l):.2f} | "
                         f"{word.get(claude_toxic.get(l['id']), '—')} | {claude.get(l['id']) or '—'} |")
    lines += [
        "",
        "## Checks",
        "",
        "- Above 0.5, the first-token probabilities give the same answer as the eval runs on this many of the 1,000 `E1000` lines: "
        + ", ".join(f"{m} {a}" for m, a in agree.items()) + ". Odds within 0.001 of 0.5 are a tie in bf16, which greedy decoding broke as "
        "`nontoxic`. So a line counts as above a threshold only when it is above it by 0.001 or more.",
        "- The least probability any model put on its two answer words, across every line: "
        + ", ".join(f"{m} {v:.3f}" for m, v in mass.items()) + ".",
        f"- Claude answered {len(claude)} lines; {sum(v is None for v in claude.values())} replies weren't `yes` or `no`.",
    ]
    (ROOT / "results" / "sarcasm_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["qwen", "claude", "report"])
    part = ap.parse_args().part
    return {"qwen": run_qwen, "claude": run_claude, "report": report}[part]()


if __name__ == "__main__":
    sys.exit(main())
