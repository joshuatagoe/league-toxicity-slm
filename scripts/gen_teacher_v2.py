"""Generate the v2 teacher examples for the diagnosed failures, then gate them.

Usage: python -I scripts/gen_teacher_v2.py

v1 failed the pressure set in two ways: it flags harmless lines that contain insult
words (quotes, reports, complaints about the game), and it misses hostility with no
insult words (sarcasm, taunts, "diff" put-downs, polite put-downs). The teacher writes
lines for each category; the label comes from the category, under docs/LABELING_RULES.md.

Gate: normalize and deduplicate, keep 2-120 characters, drop anything equal to or close
to an eval or pressure message, then a critic from the other model family checks each line
against the written rules. Lines the critic disputes are kept aside for the author.

Writes data/sft/teacher_v2.jsonl (kept), data/sft/teacher_v2_disputed.jsonl and
data/sft/teacher_v2_meta.json.
"""
import difflib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from providers import _call_anthropic, _call_openai  # noqa: E402
from run_ceiling import load_dotenv  # noqa: E402

TEACHER = "claude-opus-5-5"
CRITIC = "gpt-5.6-sol"
SEED = 2026
SETTINGS_PER_CATEGORY = 4
PER_CALL = 15

CATEGORIES = {
    # label 1: hostile with no insult words (v1 missed these)
    "sarcasm_at_teammate": (1, "Sarcastic praise that mocks a teammate's bad play, e.g. after they die, miss a skillshot, steal a kill or fall behind. Polite words, hostile meaning."),
    # Descriptions avoid the pressure set's own phrasings, so training can't copy them.
    "taunt_enemy": (1, "Taunting or gloating at the enemy team or an enemy laner after winning a fight, a lane or the game."),
    "diff_putdown": (1, "Put-downs that compare a player unfavourably with their opponent using lane or role shorthand, aimed at a teammate or the enemy."),
    "polite_putdown": (1, "Indirect or polite put-downs telling a player they are bad at the game, without using any insult words."),
    "disguised_insult": (1, "Insults aimed at a player but hidden by odd spelling, leetspeak, spacing or abbreviations. No slurs."),
    # label 0: harmless lines containing words that look hostile (v1 flagged these)
    "quote_or_report": (0, "A player quoting, describing or reporting someone else's insult or bad behaviour, or asking the team to report someone, without insulting anyone themselves."),
    "game_complaint": (0, "Complaints about a champion, item, patch, lag, matchmaking or the game itself, including swearing at the game, not at any player."),
    "self_criticism": (0, "A player criticising themselves, sometimes harshly or with swearing: admitting they fed, played badly or should uninstall."),
    "harmless_trigger_words": (0, "Normal game talk that happens to use words like noob, kill, feed, report, trash, ez, ff or die in a harmless way: a beginner's question, a warning not to give the enemy kills, calling a target to focus."),
    "feed_callout": (0, "Neutral status callouts about who is fed or behind, with no insult or mockery, given as information for the team."),
    "friendly_banter": (0, "Friendly teasing or banter that nobody would take as offensive: laughing at a funny misplay together, joking with a duo, praising the enemy."),
    "stop_the_flame": (0, "A player telling someone who is flaming to stop or calm down, sometimes bluntly, without insulting them back."),
}
SETTINGS = [
    "right after a teammate died to a gank", "after losing a teamfight at dragon", "in lane during the first 10 minutes",
    "after the enemy team surrendered", "after a teammate missed an important skillshot", "late game near baron",
    "in the post-game lobby", "after a long death timer", "while the team argues about surrendering", "after a lag spike",
]

GEN_PROMPT = """You are generating training data for a League of Legends chat moderation model. The data teaches a small classifier to recognise toxic chat and to leave harmless chat alone; this is a standard content-moderation workflow.

Write exactly {n} different in-game chat messages of this kind:
CATEGORY: {category}
DESCRIPTION: {description}
SETTING: {setting}

Style: real solo-queue League chat. Mostly 2-10 words, usually lowercase, with abbreviations and typos (gj, ty, wp, jg, adc, supp, bot, mid, ff, gg), and champion names. No hashtags, no emoji, no quotation marks around the message, no slurs. Vary wording heavily; no two alike. For reference, real messages look like these (do not copy them): {style}

Return ONLY a JSON array of strings."""

CRITIC_PROMPT = """You check League of Legends chat lines against a labelling rulebook. Apply the rulebook exactly, not your own standard.

RULEBOOK:
{rules}

For each numbered line below, decide "toxic" or "nontoxic" under the rulebook. Return ONLY a JSON array of objects like {{"i": 1, "label": "toxic"}}, one per line, in order.

{lines}"""


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def parse_array(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    rng = random.Random(SEED)
    train = [json.loads(l) for l in (ROOT / "data" / "sft" / "train_v1.jsonl").open(encoding="utf-8")]
    style_pool = [r["prompt"][1]["content"] for r in train if r["label"] == 0 and 4 <= len(r["prompt"][1]["content"]) <= 40]

    jobs = []
    for cat, (label, desc) in CATEGORIES.items():
        for setting in rng.sample(SETTINGS, SETTINGS_PER_CATEGORY):
            style = "; ".join(rng.sample(style_pool, 6))
            jobs.append((cat, label, GEN_PROMPT.format(n=PER_CALL, category=cat, description=desc, setting=setting, style=style)))

    def generate(job):
        cat, label, prompt = job
        reply = _call_anthropic(TEACHER, prompt, "low")
        return [{"message": str(m).strip(), "label": label, "category": cat} for m in parse_array(reply.text) if isinstance(m, str)], reply

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(generate, jobs))
    raw = [row for rows, _ in results for row in rows]
    tokens = [(r.input_tokens, r.output_tokens) for _, r in results]
    print(f"generated {len(raw)} lines from {len(jobs)} teacher calls")

    # Gate 1-3: dedupe, length, leakage against every eval and pressure message.
    held_out = set()
    for path in (ROOT / "data" / "eval" / "FULL.jsonl", ROOT / "data" / "pressure" / "PRESSURE.jsonl"):
        held_out |= {norm(json.loads(l)["message"]) for l in path.open(encoding="utf-8") if l.strip()}
    pressure = [norm(json.loads(l)["message"]) for l in (ROOT / "data" / "pressure" / "PRESSURE.jsonl").open(encoding="utf-8")]
    train_keys = {norm(r["prompt"][1]["content"]) for r in train}
    seen, gated, dropped = set(), [], {"duplicate": 0, "length": 0, "in_eval": 0, "near_pressure": 0, "in_train": 0}
    for row in raw:
        key = norm(row["message"])
        if key in seen:
            dropped["duplicate"] += 1
        elif not 2 <= len(row["message"]) <= 120:
            dropped["length"] += 1
        elif key in held_out:
            dropped["in_eval"] += 1
        elif any(difflib.SequenceMatcher(None, key, p).ratio() >= 0.8 for p in pressure):
            dropped["near_pressure"] += 1
        elif key in train_keys:
            dropped["in_train"] += 1
        else:
            seen.add(key)
            gated.append(row)
    print(f"after dedupe, length and leakage checks: {len(gated)} (dropped {dropped})")

    # Gate 4: critic from the other model family, applying the written rules.
    rules = (ROOT / "docs" / "LABELING_RULES.md").read_text(encoding="utf-8")
    batches = [gated[i : i + 20] for i in range(0, len(gated), 20)]

    def critique(batch):
        lines = "\n".join(f"{i + 1}. {row['message']}" for i, row in enumerate(batch))
        reply = _call_openai(CRITIC, CRITIC_PROMPT.format(rules=rules, lines=lines), "low")
        verdicts = {int(v["i"]): v.get("label") for v in parse_array(reply.text) if isinstance(v, dict) and "i" in v}
        return [verdicts.get(i + 1) for i in range(len(batch))], reply

    with ThreadPoolExecutor(8) as pool:
        critiques = list(pool.map(critique, batches))
    kept, disputed = [], []
    for batch, (verdicts, _) in zip(batches, critiques):
        for row, verdict in zip(batch, verdicts):
            row["critic"] = verdict
            (kept if verdict == ("toxic" if row["label"] else "nontoxic") else disputed).append(row)
    critic_tokens = [(r.input_tokens, r.output_tokens) for _, r in critiques]

    out = ROOT / "data" / "sft"
    for name, rows in (("teacher_v2.jsonl", kept), ("teacher_v2_disputed.jsonl", disputed)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cost = sum(i * 4.0 + o * 20.0 for i, o in tokens) / 1e6 + sum(i * 5.0 + o * 30.0 for i, o in critic_tokens) / 1e6
    meta = {
        "teacher": TEACHER, "critic": CRITIC, "seed": SEED, "calls": len(jobs), "generated": len(raw),
        "dropped": dropped, "kept": len(kept), "disputed": len(disputed), "cost_usd_list_price": round(cost, 2),
        "kept_by_category": {c: sum(r["category"] == c for r in kept) for c in CATEGORIES},
        "disputed_by_category": {c: sum(r["category"] == c for r in disputed) for c in CATEGORIES},
    }
    (out / "teacher_v2_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
