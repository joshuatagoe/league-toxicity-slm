"""Read an eval set in whatever shape it arrives: ours, or a grader's held-out file.

Accepts JSON Lines, a JSON list, CSV or TSV with a header row, or plain text with one
message per line. The message is the first of MESSAGE_KEYS a row has (or the last user
turn of a "messages" list); the label is the first of LABEL_KEYS, given as 1/0,
true/false or toxic/nontoxic. A row without a label is still labelled by the model,
just not scored. A label that can't be read stops the run rather than being guessed.
"""
import csv
import json
from pathlib import Path

MESSAGE_KEYS = ("message", "text", "input", "prompt", "content", "chat", "line", "scenario", "query")
LABEL_KEYS = ("label", "gold", "expected", "expected_label", "target", "toxic", "is_toxic")
TOXIC = {"1", "1.0", "true", "yes", "toxic"}
NONTOXIC = {"0", "0.0", "false", "no", "nontoxic", "non-toxic", "non toxic", "not toxic", "not_toxic"}


def parse_gold(value) -> int | None:
    """1 = toxic, 0 = not toxic, None = no label given."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in TOXIC:
        return 1
    if s in NONTOXIC:
        return 0
    raise ValueError(f"can't read label {value!r}: use 1/0, true/false or toxic/nontoxic")


def _message(row: dict):
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in MESSAGE_KEYS:
        if isinstance(lowered.get(key), str):  # an empty message is still a row (the test split has one)
            return lowered[key]
    turns = lowered.get("messages")
    if isinstance(turns, list):
        users = [t.get("content") for t in turns if isinstance(t, dict) and t.get("role") == "user"]
        if users and isinstance(users[-1], str):
            return users[-1]
    return None


def _label(row: dict):
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in LABEL_KEYS:
        if key in lowered:
            return lowered[key]
    return None


def _raw_rows(path: Path) -> list:
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return list(csv.DictReader(text.splitlines(), delimiter="\t" if suffix == ".tsv" else ","))
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):  # {"rows": [...]} or similar: take the one list inside
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise ValueError(f"{path}: expected a list of rows, or an object holding one list")
            data = lists[0]
        return data
    if suffix == ".txt":
        return [{"message": line} for line in text.splitlines() if line.strip()]
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # .jsonl and anything else


def read_eval_set(path: str | Path) -> list[dict]:
    """Rows as {"id", "message", "label"}, with label None when the file gives none."""
    path = Path(path)
    rows = []
    for i, raw in enumerate(_raw_rows(path), start=1):
        if isinstance(raw, str):
            raw = {"message": raw}
        message = _message(raw)
        if message is None:
            raise ValueError(f"{path} row {i}: no message field; looked for {', '.join(MESSAGE_KEYS)} (found {', '.join(map(str, raw))})")
        try:
            gold = parse_gold(_label(raw))
        except ValueError as e:
            raise ValueError(f"{path} row {i}: {e}") from None
        row_id = raw.get("id", raw.get("ID"))
        rows.append({"id": str(row_id) if row_id not in (None, "") else str(i), "message": message, "label": gold})
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: ids repeat; give each row a unique id or none at all")
    return rows
