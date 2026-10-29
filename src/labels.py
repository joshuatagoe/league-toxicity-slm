"""Turn a model's reply into a label: 1 = toxic, 0 = nontoxic, None = unparseable."""
import re

_LABEL_FIELD = re.compile(r'"label"\s*:\s*"([^"]*)"', re.I)
_NEGATED = re.compile(r"\b(?:non|not)\s*[-_]?\s*toxic\b")
_LABEL = re.compile(r"\b(nontoxic|toxic)\b")


def parse_label(raw: str, structured: bool = False) -> int | None:
    if not raw:
        return None
    if structured:
        # The structured prompt puts reasoning before the label, and the reasoning
        # may itself say "toxic", so only the label field counts.
        m = _LABEL_FIELD.search(raw)
        if not m:
            return None
        raw = m.group(1)
    s = raw.lower().replace("`", " ").replace('"', " ").replace("'", " ")
    s = _NEGATED.sub("nontoxic", s)
    m = _LABEL.search(s)
    if not m:
        return None
    return int(m.group(1) == "toxic")
