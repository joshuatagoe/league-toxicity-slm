"""Build the training and eval data as a Hugging Face dataset, and optionally publish it.

Usage (from the repo root):
  python -I scripts/hf_dataset.py           # build only, into build/dataset
  .venv-train/Scripts/python -I scripts/hf_dataset.py --push    # build, then create or update <user>/league-toxicity-chat

Every file the model was trained or scored on is copied unchanged (sft/, teacher/, eval/, pressure/),
so its SHA-256 matches the repo's manifests; the build checks that. The training files are also
written flat (flat/: id, message, label, source, category) for the Hub's viewer and load_dataset().
docs/DATASET.md is the card, with its repo links pointed at GitHub. --push reads HF_TOKEN from .env
and needs the code committed and on GitHub, since the card cites the commit.
"""
import argparse
import hashlib
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "build" / "dataset"
REPO = "league-toxicity-chat"
GITHUB = "https://github.com/joshuatagoe/league-toxicity-slm"
COPIES = {  # path here: path in the repo
    "sft/train_v1.jsonl": "data/sft/train_v1.jsonl",
    "sft/val.jsonl": "data/sft/val.jsonl",
    "sft/train_v2.jsonl": "data/sft/train_v2.jsonl",
    "sft/train_v2_small.jsonl": "data/sft/train_v2_small.jsonl",
    "sft/train_v3.jsonl": "data/sft/train_v3.jsonl",
    "sft/manifest.json": "data/sft/manifest.json",
    "teacher/teacher_v2.jsonl": "data/sft/teacher_v2.jsonl",
    "teacher/teacher_v2_disputed.jsonl": "data/sft/teacher_v2_disputed.jsonl",
    "teacher/teacher_v2_meta.json": "data/sft/teacher_v2_meta.json",
    "eval/E1000.jsonl": "data/eval/E1000.jsonl",
    "eval/FULL.jsonl": "data/eval/FULL.jsonl",
    "eval/manifest.json": "data/manifest.json",
    "pressure/PRESSURE.jsonl": "data/pressure/PRESSURE.jsonl",
}
FLAT = ["train_v1", "val", "train_v2"]
YAML = """---
license: mit
language:
  - en
pretty_name: League of Legends chat toxicity (L2DTnH splits plus teacher lines)
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
tags:
  - toxicity
  - content-moderation
  - league-of-legends
  - gaming
  - synthetic
configs:
  - config_name: train_v2
    default: true
    data_files:
      - split: train
        path: flat/train_v2.jsonl
      - split: validation
        path: flat/val.jsonl
  - config_name: train_v1
    data_files:
      - split: train
        path: flat/train_v1.jsonl
      - split: validation
        path: flat/val.jsonl
  - config_name: teacher_v2
    data_files:
      - split: kept
        path: teacher/teacher_v2.jsonl
      - split: disputed
        path: teacher/teacher_v2_disputed.jsonl
  - config_name: test
    data_files:
      - split: e1000
        path: eval/E1000.jsonl
      - split: full
        path: eval/FULL.jsonl
  - config_name: pressure
    data_files:
      - split: test
        path: pressure/PRESSURE.jsonl
---
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes() -> None:
    """The copies must be the exact files the manifests (and the training logs) name."""
    sft = json.loads((ROOT / "data/sft/manifest.json").read_text())["sha256"]
    ev = json.loads((ROOT / "data/manifest.json").read_text())["eval"]
    for name, digest in {**{f"data/sft/{k}": v for k, v in sft.items()}, **{f"data/eval/{k}": v for k, v in ev.items()}}.items():
        if sha256(ROOT / name) != digest:
            sys.exit(f"{name} doesn't match its manifest hash; rebuild the data before publishing")


def flat(name: str) -> list[dict]:
    rows = []
    for line in (ROOT / "data/sft" / f"{name}.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            teacher = r["id"].startswith("t2-")
            rows.append({"id": r["id"], "message": r["prompt"][1]["content"], "label": r["label"],
                         "source": "teacher" if teacher else "l2dtnh", "category": r.get("category")})
    return rows


def card(ref: str, code_commit: str | None) -> str:
    body = (ROOT / "docs" / "DATASET.md").read_text(encoding="utf-8")

    def link(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"[a-z]+:|#", target):
            return m.group(0)
        path, _, anchor = target.partition("#")
        full = posixpath.normpath(posixpath.join("docs", path))
        return f"{m.group(1)}({GITHUB}/blob/{ref}/{full}{'#' + anchor if anchor else ''})"

    body = re.sub(r"(\[[^\]]*\])\(([^)\s]+)\)", link, body)
    body = re.sub(r"\n+For the final submission the training and eval files will be published.*\n?", "\n", body)
    built = f"commit [`{code_commit[:7]}`]({GITHUB}/tree/{code_commit}) of " if code_commit else ""
    head = f"""
> Built from {built}[league-toxicity-slm]({GITHUB}). The model trained on it is
> [Juchen/league-toxicity-qwen3-1.7b](https://huggingface.co/Juchen/league-toxicity-qwen3-1.7b).

## Files here

| Here | In the GitHub repo | What |
| --- | --- | --- |
| `sft/` | `data/sft/` | The exact training files, chat format, as trained (hashes in `sft/manifest.json`) |
| `flat/` | (built from `sft/`) | The same rows as `id, message, label, source, category`, for the viewer and `load_dataset` |
| `teacher/` | `data/sft/teacher_v2*` | The 676 kept teacher lines, the 40 the critic disputed, and the generation log |
| `eval/` | `data/eval/` | The test split (`FULL`) and its first 1,000 rows (`E1000`), hashes in `eval/manifest.json` |
| `pressure/` | `data/pressure/` | The 40-line pressure set |

```python
from datasets import load_dataset
train = load_dataset("Juchen/{REPO}", "train_v2")       # 11,595 training rows + 1,213 validation
test = load_dataset("Juchen/{REPO}", "test", split="e1000")
```

"""
    rest = body.split("\n", 1)[1]  # DATASET.md's own title reads oddly on the Hub
    return YAML + "# League of Legends chat toxicity: training and eval data\n" + head + rest.lstrip("\n")


def build(ref: str, code_commit: str | None) -> None:
    check_hashes()
    if OUT.exists():
        shutil.rmtree(OUT)
    for here, there in COPIES.items():
        (OUT / here).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / there, OUT / here)
    (OUT / "flat").mkdir()
    for name in FLAT:
        with (OUT / "flat" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in flat(name):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "README.md").write_text(card(ref, code_commit), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="publish the dataset (public)")
    args = ap.parse_args()
    if not args.push:
        build("main", None)
        print(f"built {OUT.relative_to(ROOT)}: " + ", ".join(sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())))
        return 0

    from huggingface_hub import HfApi
    from push_hf import git, token

    code_commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("commit your changes first: the card cites the code commit")
    if not git("branch", "-r", "--contains", code_commit):
        sys.exit("push first: the card links to the code commit on GitHub")
    build(code_commit, code_commit)
    api = HfApi(token=token())
    repo_id = f"{api.whoami()['name']}/{REPO}"
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(OUT),
                             commit_message=f"Data from code commit {code_commit[:7]}", delete_patterns=["*"])
    print(f"dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"dataset commit: {info.oid}")
    print(f"code commit: {code_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
