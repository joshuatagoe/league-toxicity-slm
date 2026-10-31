"""The held-out reader: the shapes a grader's file might come in.

Usage: python -I tests/test_evalsets.py   (pytest also collects it, if installed)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evalsets import read_eval_set  # noqa: E402


def write(name: str, text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text, encoding="utf-8")
    return path


def raises(path: Path, words: str) -> None:
    try:
        read_eval_set(path)
    except ValueError as e:
        assert words in str(e), str(e)
        return
    raise AssertionError(f"{path.name} should have been refused")


def test_our_eval_sets_read_unchanged():
    for rel in ["data/eval/E1000.jsonl", "data/pressure/PRESSURE.jsonl", "data/eval/FULL.jsonl"]:
        old = [json.loads(l) for l in (ROOT / rel).open(encoding="utf-8") if l.strip()]
        new = read_eval_set(ROOT / rel)
        assert [(o["id"], o["message"], o["label"]) for o in old] == [(n["id"], n["message"], n["label"]) for n in new]


def test_csv_with_text_column_and_word_labels():
    rows = read_eval_set(write("heldout.csv", "text,label\ngg ez,toxic\nward drag pls,Non-Toxic\n\"go next, ff\",\n"))
    assert rows == [
        {"id": "1", "message": "gg ez", "label": 1},
        {"id": "2", "message": "ward drag pls", "label": 0},
        {"id": "3", "message": "go next, ff", "label": None},
    ]


def test_tsv_and_boolean_labels():
    rows = read_eval_set(write("heldout.tsv", "ID\tInput\tis_toxic\na1\tuninstall\tTrue\na2\tnice ult\tfalse\n"))
    assert [(r["id"], r["message"], r["label"]) for r in rows] == [("a1", "uninstall", 1), ("a2", "nice ult", 0)]


def test_jsonl_with_prompt_and_expected():
    rows = read_eval_set(write("heldout.jsonl", '{"prompt": "jg diff", "expected": "toxic"}\n\n{"prompt": "my bad", "expected": 0}\n'))
    assert [(r["message"], r["label"]) for r in rows] == [("jg diff", 1), ("my bad", 0)]


def test_json_list_object_and_chat_messages():
    data = {"scenarios": [
        {"id": 7, "messages": [{"role": "system", "content": "ignored"}, {"role": "user", "content": "report bot"}], "label": 0},
        "just a bare string",
    ]}
    rows = read_eval_set(write("heldout.json", json.dumps(data)))
    assert rows == [{"id": "7", "message": "report bot", "label": 0}, {"id": "2", "message": "just a bare string", "label": None}]


def test_plain_text_has_no_labels():
    rows = read_eval_set(write("heldout.txt", "gg ez\n\nff 15\n"))
    assert [(r["message"], r["label"]) for r in rows] == [("gg ez", None), ("ff 15", None)]


def test_unreadable_label_is_refused_not_guessed():
    raises(write("bad.csv", "text,label\ngg ez,maybe\n"), "row 1: can't read label 'maybe'")


def test_missing_message_field_is_refused():
    raises(write("bad.jsonl", '{"utterance": "gg ez", "label": 1}\n'), "no message field")


def test_repeated_ids_are_refused():
    raises(write("bad.jsonl", '{"id": 1, "text": "a"}\n{"id": 1, "text": "b"}\n'), "ids repeat")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
