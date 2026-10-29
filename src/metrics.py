"""Precision, recall and F1 for the toxic class, with bootstrap intervals.

A reply that failed (unparseable, refused, truncated or errored) counts as a wrong
answer, so every row of the eval set is scored. The paper dropped such rows instead.
"""
import random
from collections import Counter


def confusion(records: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    failures = Counter()
    for r in records:
        gold, pred = r["gold"], r["pred"]
        if pred is None:
            failures[r["status"]] += 1
            pred = 1 - gold
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1:
            fp += 1
        elif gold == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "failures": dict(failures)}


def scores(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = tp + fp + tn + fn
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / n if n else 0.0, "n": n}


def bootstrap(records: list[dict], rounds: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals for precision, recall and F1."""
    rng = random.Random(seed)
    samples = {"precision": [], "recall": [], "f1": []}
    for _ in range(rounds):
        resample = [records[rng.randrange(len(records))] for _ in records]
        s = scores(confusion(resample))
        for k in samples:
            samples[k].append(s[k])
    out = {}
    for k, values in samples.items():
        values.sort()
        out[k] = (values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1])
    return out
