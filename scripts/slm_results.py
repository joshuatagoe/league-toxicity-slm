"""Build the small-model results: base vs tuned, paired differences, the pressure set by
kind, the data-efficiency curve (table and SVG chart) and the training logs.

Usage: python -I scripts/slm_results.py

Reads runs/*.jsonl (bf16 scores, full eval sets) and results/train_logs/*.json (copied
from each model folder, which git ignores). Writes results/slm_results.md and
results/data_efficiency.svg; nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import bootstrap, confusion, paired, scores  # noqa: E402

BAR = {"precision": 0.80, "f1": 0.75}  # set in the README before any run
CURVE = ["qwen3-1.7b-v1-n1364", "qwen3-1.7b-v1-n2729", "qwen3-1.7b-v1-n5459", "qwen3-1.7b-v1-n10919"]
SEEDS = (3407, 42)
SEED2 = "-seed42"  # the same runs with training seed 42 instead of 3407 (train/run_seed2.sh)
CURVE2 = [m + SEED2 for m in CURVE]
TUNED = CURVE + ["qwen3-1.7b-v2-small", "qwen3-1.7b-v2"] + CURVE2 + ["qwen3-1.7b-v2" + SEED2, "qwen3-1.7b-v3"]
FRONTIER = [("claude-opus-5-5", "zero_shot_p1"), ("gpt-5.6-sol", "few_shot")]  # each family's best F1 cell
BERT = "paper-bert"  # the L2DTnH paper's fine-tuned BERT, scored by src/run_paper_bert.py
PAIRS = [  # (tuned, compared with, why)
    ("qwen3-1.7b-v1-n10919", "qwen3-1.7b-base", "v1 vs untrained"),
    ("qwen3-1.7b-v2", "qwen3-1.7b-base", "v2 vs untrained"),
    ("qwen3-1.7b-v1-n10919", "qwen3-1.7b-v1-n5459", "v1: all 10,919 rows vs half"),
    ("qwen3-1.7b-v2", "qwen3-1.7b-v1-n10919", "v2 vs v1: the data fix"),
    ("qwen3-1.7b-v2-small", "qwen3-1.7b-v1-n2729", "v2-small vs v1 at 2,729 rows: the data fix on a small set"),
    ("qwen3-1.7b-v2-seed42", "qwen3-1.7b-v1-n10919-seed42", "v2 vs v1, both trained with seed 42: the data fix again"),
    ("qwen3-1.7b-v3", "qwen3-1.7b-v2", "v3 vs v2: the teacher lines at 4x the dose"),
    ("qwen3-1.7b-v1-n10919-seed42", "qwen3-1.7b-v1-n5459-seed42", "v1 seed 42: all 10,919 rows vs half"),
    ("qwen3-1.7b-v1-n10919-seed42", "qwen3-1.7b-v1-n10919", "v1, 10,919 rows: seed 42 vs seed 3407 (training noise)"),
    ("qwen3-1.7b-v2-seed42", "qwen3-1.7b-v2", "v2: seed 42 vs seed 3407 (training noise)"),
    ("qwen3-1.7b-v2", BERT, "v2 vs the paper's fine-tuned BERT (BERT's E1000 is flattered, see †)"),
]
NAMES = {
    "qwen3-1.7b-base": "Qwen3-1.7B, untrained",
    "qwen3-1.7b-v1-n1364": "v1, 1,364 rows",
    "qwen3-1.7b-v1-n2729": "v1, 2,729 rows",
    "qwen3-1.7b-v1-n5459": "v1, 5,459 rows",
    "qwen3-1.7b-v1-n10919": "v1, 10,919 rows",
    "qwen3-1.7b-v2-small": "v2-small, 2,729 + 676 teacher rows",
    "qwen3-1.7b-v2": "v2, 10,919 + 676 teacher rows",
    "qwen3-1.7b-v3": "v3, 10,919 + 676 teacher rows x 4",
    **{m + SEED2: NAMES_SEED2 for m, NAMES_SEED2 in [
        ("qwen3-1.7b-v1-n1364", "v1, 1,364 rows, seed 42"), ("qwen3-1.7b-v1-n2729", "v1, 2,729 rows, seed 42"),
        ("qwen3-1.7b-v1-n5459", "v1, 5,459 rows, seed 42"), ("qwen3-1.7b-v1-n10919", "v1, 10,919 rows, seed 42"),
        ("qwen3-1.7b-v2", "v2, 10,919 + 676 teacher rows, seed 42")]},
    "claude-opus-5-5": "Claude Opus 5.5, zero-shot",
    "gpt-5.6-sol": "GPT-5.6 Sol, few-shot",
    BERT: "Paper's fine-tuned BERT†",
}


def run(cell: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / "runs" / f"{cell}.jsonl").open(encoding="utf-8") if l.strip()]


def slm(model: str, eval_set: str) -> list[dict]:
    return run(f"{model}__slm__na__{eval_set}")


def correct(records: list[dict]) -> int:
    return sum(r["pred"] == r["gold"] for r in records)


def chart(points: list[tuple], base_f1: float, base_pressure: float) -> str:
    """Line chart of F1 and pressure accuracy against training rows, log2 x axis.

    points: (rows, [F1 per seed], [accuracy per seed]). The line joins the seed means; each seed is
    a dot, filled for the first and open for the second. Dots are nudged sideways, F1 left and
    robustness right, so equal values stay visible.
    """
    import math

    w, h, left, right, top, bottom = 680, 400, 56, 196, 76, 48
    x0, x1 = math.log2(1000), math.log2(14000)
    sx = lambda n: left + (math.log2(n) - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    sy = lambda v: top + (1 - v) * (h - top - bottom)  # noqa: E731
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
        "<style>",
        ".bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.muted{fill:#898781}.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}"
        ".ref{stroke:#898781}.s1{stroke:#2a78d6}.f1{fill:#2a78d6}.s2{stroke:#eb6834}.f2{fill:#eb6834}.ring{stroke:#fcfcfb}.open{fill:#fcfcfb}",
        "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}"
        ".s1{stroke:#3987e5}.f1{fill:#3987e5}.s2{stroke:#d95926}.f2{fill:#d95926}.ring{stroke:#1a1a19}.open{fill:#1a1a19}}",
        "</style>",
        f'<rect class="bg" width="{w}" height="{h}" rx="8"/>',
        f'<text class="ink" x="{left}" y="22" font-size="14" font-weight="600">Data-efficiency curve: Qwen3-1.7B v1, nested subsets of the expert-labelled data</text>',
        f'<line class="s1" x1="{left}" y1="40" x2="{left + 18}" y2="40" stroke-width="2"/><circle class="f1" cx="{left + 9}" cy="40" r="4"/>'
        f'<text class="ink2" x="{left + 24}" y="44">Spec adherence: F1 on E1000</text>',
        f'<line class="s2" x1="{left + 210}" y1="40" x2="{left + 228}" y2="40" stroke-width="2"/><circle class="f2" cx="{left + 219}" cy="40" r="4"/>'
        f'<text class="ink2" x="{left + 234}" y="44">Robustness: accuracy on the 40 pressure lines</text>',
        f'<circle class="f1" cx="{left + 5}" cy="58" r="4"/><circle class="s1 open" cx="{left + 17}" cy="58" r="3.5" stroke-width="1.5"/>'
        f'<text class="ink2" x="{left + 28}" y="62">Dots: training seed 3407 (filled) and 42 (open). Line: their mean.</text>',
    ]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke-width="1"/>')
        out.append(f'<text class="muted" x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke-width="1"/>')
    for n, *_ in points:
        out.append(f'<text class="muted" x="{sx(n):.1f}" y="{h - bottom + 18}" text-anchor="middle">{n:,}</text>')
    out.append(f'<text class="muted" x="{(left + w - right) / 2:.1f}" y="{h - 8}" text-anchor="middle">Training rows (log scale)</text>')
    refs = [(BAR["f1"], "F1 bar 0.75", "6 4"), (base_f1, f"untrained F1 {base_f1:.3f}", "2 3"), (base_pressure, f"untrained robustness {base_pressure:.3f}", "2 3")]
    for v, label, dash in refs:
        out.append(f'<line class="ref" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke-width="1" stroke-dasharray="{dash}"/>')
        out.append(f'<text class="ink2" x="{w - right + 8}" y="{sy(v) + 4:.1f}">{label}</text>')
    for cls, i, name in (("1", 1, "F1"), ("2", 2, "robustness")):
        path = " ".join(f"{'M' if k == 0 else 'L'}{sx(p[0]):.1f},{sy(sum(p[i]) / len(p[i])):.1f}" for k, p in enumerate(points))
        out.append(f'<path class="s{cls}" d="{path}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
        for p in points:
            for j, (v, seed) in enumerate(zip(p[i], SEEDS)):
                x = sx(p[0]) + (i - 1.5) * 12 + (j - 0.5) * 9
                look = f'class="f{cls} ring" r="4.5" stroke-width="2"' if j == 0 else f'class="s{cls} open" r="4" stroke-width="1.75"'
                out.append(f'<circle {look} cx="{x:.1f}" cy="{sy(v):.1f}"><title>{p[0]:,} rows, seed {seed}: {name} {v:.3f}</title></circle>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logs = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results" / "train_logs").glob("*.json")}
    models = ["qwen3-1.7b-base"] + TUNED
    spec = {m: slm(m, "E1000") for m in models}
    robust = {m: slm(m, "PRESSURE") for m in models}
    for model, strategy in FRONTIER:
        spec[model] = run(f"{model}__{strategy}__low__E1000")
        robust[model] = run(f"{model}__{strategy}__low__PRESSURE")
    spec[BERT], robust[BERT] = run(f"{BERT}__classifier__na__E1000"), run(f"{BERT}__classifier__na__PRESSURE")
    bert = json.loads((ROOT / "runs" / f"{BERT}__classifier__na__E1000.meta.json").read_text())

    lines = [
        "# Small model results",
        "",
        "Generated by `scripts/slm_results.py` from `runs/` and `results/train_logs/`. Every small-model run here is scored in bf16 on the full eval set. "
        "`E1000` is the first 1,000 L2DTnH test messages, scored against the expert annotators (102 toxic). "
        "The pressure set is 40 hard lines, 4 for each of 10 kinds, labelled by the author under [docs/LABELING_RULES.md](../docs/LABELING_RULES.md) (11 toxic, 29 not). "
        f"Bar, set before any run: precision ≥ {BAR['precision']:.2f} and F1 ≥ {BAR['f1']:.2f} on `E1000`. "
        "A reply counts only if it is exactly `toxic` or `nontoxic`; anything else is scored wrong.",
        "",
        "## Base vs tuned",
        "",
        "| Model | Training rows | Precision | Recall | F1 (95% CI) | Pressure set | Meets bar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in models + [f[0] for f in FRONTIER] + [BERT]:
        s, ci = scores(confusion(spec[m])), bootstrap(spec[m])["f1"]
        rows = f"{logs[m]['n']:,}" if m in logs else {"qwen3-1.7b-base": "0", BERT: f"{bert['train_rows']:,} (full fine-tune)"}.get(m, "prompted")
        bar = "yes" if s["precision"] >= BAR["precision"] and s["f1"] >= BAR["f1"] else "no"
        lines.append(f"| {NAMES[m]} | {rows} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} ({ci[0]:.2f}–{ci[1]:.2f}) | "
                     f"{correct(robust[m])}/40 | {bar} |")
    default = scores(confusion([{**r, "pred": int(float(r["raw"]) >= 0.5)} for r in spec[BERT]]))
    lines += [
        "",
        "† The L2DTnH authors' own fine-tune of `unitary/toxic-bert` (Ave et al., ESORICS 2026), pinned to the data's commit and "
        f"scored by `src/run_paper_bert.py` in fp32 on the CPU, with the paper's threshold ({bert['threshold']}) and {bert['max_length']}-token limit. "
        "Its training script used the test split as its validation set: it kept the epoch with the lowest test loss and chose the "
        "threshold on the test split. `E1000` is the first 1,000 rows of that split, so its `E1000` scores are flattered. At the default "
        f"0.50 threshold it scores precision {default['precision']:.3f} and F1 {default['f1']:.3f} on `E1000`; the epoch choice can't be undone "
        "without retraining. Its training never saw the pressure set, so the pressure column is the fair comparison.",
    ]

    lines += [
        "",
        "## Paired differences",
        "",
        "Each interval resamples the same messages for both models (1,000 rounds), so it measures the difference itself. "
        "An interval that crosses 0 means the data can't tell the two apart.",
        "",
        "| Comparison | F1 change on E1000 (95% CI) | Precision change (95% CI) | Pressure set change (95% CI, accuracy) |",
        "| --- | --- | --- | --- |",
    ]
    for a, b, why in PAIRS:
        f1, pr = paired(spec[a], spec[b], "f1"), paired(spec[a], spec[b], "precision")
        acc = paired(robust[a], robust[b], "accuracy")
        d = correct(robust[a]) - correct(robust[b])
        lines.append(f"| {why} | {f1[0]:+.3f} ({f1[1]:+.2f} to {f1[2]:+.2f}) | {pr[0]:+.3f} ({pr[1]:+.2f} to {pr[2]:+.2f}) | "
                     f"{d:+d} line{'' if abs(d) == 1 else 's'} ({acc[1]:+.2f} to {acc[2]:+.2f}) |")

    kinds = list(dict.fromkeys(json.loads(l)["kind"] for l in (ROOT / "data" / "pressure" / "PRESSURE.jsonl").open(encoding="utf-8")))
    shown = ["qwen3-1.7b-base", "qwen3-1.7b-v1-n10919", "qwen3-1.7b-v2", "qwen3-1.7b-v3", "claude-opus-5-5", BERT]
    pressure = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "pressure" / "PRESSURE.jsonl").open(encoding="utf-8")}
    lines += [
        "",
        "## Pressure set by kind",
        "",
        "Correct answers out of 4 per kind; the toxic count is how many of the 4 lines are toxic under the written rules.",
        "",
        "| Kind | Toxic lines | " + " | ".join(NAMES[m] for m in shown) + " |",
        "| --- | --- | " + " | ".join("---" for _ in shown) + " |",
    ]
    for k in kinds:
        ids = [i for i, p in pressure.items() if p["kind"] == k]
        cells = [str(sum(r["pred"] == r["gold"] for r in robust[m] if r["id"] in ids)) for m in shown]
        lines.append(f"| {k} | {sum(pressure[i]['label'] for i in ids)} | " + " | ".join(cells) + " |")
    lines.append("| **Toxic lines caught** | 11 | " + " | ".join(str(sum(r["pred"] == 1 == r["gold"] for r in robust[m])) for m in shown) + " |")
    lines.append("| **Harmless lines left alone** | 29 | " + " | ".join(str(sum(r["pred"] == 0 == r["gold"] for r in robust[m])) for m in shown) + " |")

    points, by_seed = [], []
    for m, m2 in zip(CURVE, CURVE2):
        seeds = [scores(confusion(spec[x])) for x in (m, m2)]
        accs = [scores(confusion(robust[x]))["accuracy"] for x in (m, m2)]
        points.append((logs[m]["n"], [s["f1"] for s in seeds], accs))
        by_seed.append((m, seeds, accs))
    base_f1 = scores(confusion(spec["qwen3-1.7b-base"]))["f1"]
    base_pressure = scores(confusion(robust["qwen3-1.7b-base"]))["accuracy"]
    (ROOT / "results" / "data_efficiency.svg").write_text(chart(points, base_f1, base_pressure), encoding="utf-8", newline="\n")
    lines += [
        "",
        "## Data-efficiency curve",
        "",
        "v1 checkpoints trained on nested prefixes of the shuffled expert-labelled training file (each subset contains the smaller ones), "
        "identical settings, two training seeds per size: 3407 and 42. Both seeds train on the same rows; the seed sets the LoRA "
        "initialisation and the batch order. Each cell gives seed 3407 / seed 42, then their mean.",
        "",
        "![Data-efficiency curve](data_efficiency.svg)",
        "",
        "| Training rows | Toxic rows | Spec adherence (F1, E1000) | Precision | Recall | Robustness (pressure accuracy) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    two = lambda a, b: f"{a:.3f} / {b:.3f} ({(a + b) / 2:.3f})"  # noqa: E731
    for m, (a, b), accs in by_seed:
        lines.append(f"| {logs[m]['n']:,} | {logs[m]['toxic']:,} | {two(a['f1'], b['f1'])} | {two(a['precision'], b['precision'])} | "
                     f"{two(a['recall'], b['recall'])} | {two(*accs)} |")

    lines += [
        "",
        "## Training logs",
        "",
        "From `results/train_logs/`: QLoRA on a 4-bit NF4 base, LoRA rank 16, alpha 16, learning rate 2e-4, cosine schedule, 1 epoch, batch 16, "
        "seed 3407 (42 for the `-seed42` runs), "
        "on one RTX 4060 laptop GPU (8 GB). Loss is the mean training loss over the run. "
        "Minutes are wall-clock time on a laptop that was doing other work too, so they don't compare well between runs.",
        "",
        "| Model | Train file | Rows | Toxic | Seed | Steps | Minutes | Mean loss | Train file SHA-256 | Code commit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in TUNED:
        g = logs[m]
        lines.append(f"| {m} | `{g['train_file']}` | {g['n']:,} | {g['toxic']:,} | {g['args']['seed']} | {g['steps']} | {g['minutes']} | {g['train_loss']:.4f} | "
                     f"`{g['train_sha256'][:12]}` | `{g['git_commit'][:7]}` |")

    (ROOT / "results" / "slm_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
