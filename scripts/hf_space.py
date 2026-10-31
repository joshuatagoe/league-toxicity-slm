"""Build the hosted demo as a Hugging Face Space (Docker, free CPU), and optionally publish it.

Usage (training environment, from the repo root):
  .venv-train/Scripts/python -I scripts/hf_space.py           # build only, into build/space
  .venv-train/Scripts/python -I scripts/hf_space.py --push    # build, then create or update <user>/league-toxicity-demo

The Space runs demo/server.py unchanged: the untrained Qwen3-1.7B and the public v2 adapter, both
at the commits in src/pins.py, downloaded into the image at build time so a cold start doesn't wait
on the Hub. It copies only the files the server reads. --push reads HF_TOKEN from .env, needs the
code committed and on GitHub (the Space cites the commit), and prints the Space URL and its commit.
"""
import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pins import BASE, BASE_REVISION, TUNED, TUNED_REVISION  # noqa: E402

OUT = ROOT / "build" / "space"
REPO = "league-toxicity-demo"
GITHUB = "https://github.com/joshuatagoe/league-toxicity-slm"
CODE = ["demo/server.py", "demo/index.html", "demo/examples.json", "results/data_efficiency.svg",
        "src/labels.py", "src/slm.py", "src/slm_prompt.py", "src/pins.py"]
# CPU builds of the same library versions as requirements-train.txt.
REQUIREMENTS = """transformers==5.22.0
peft==0.21.2
accelerate==1.15.0
huggingface-hub==2.4.0
"""
DOCKERFILE = f"""FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \\
 && pip install --no-cache-dir -r requirements.txt
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
# Bake the pinned weights into the image, so a cold start loads from disk.
RUN python -c "from huggingface_hub import snapshot_download as get; \\
get('{BASE}', revision='{BASE_REVISION}'); get('{TUNED}', revision='{TUNED_REVISION}')"
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "demo/server.py", "--host", "0.0.0.0", "--port", "7860"]
"""


def server_runs() -> list[str]:
    """The run names demo/server.py reads its results table from (its RESULT_ROWS)."""
    tree = ast.parse((ROOT / "demo" / "server.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "RESULT_ROWS":
            return [run for run, _ in ast.literal_eval(node.value)]
    raise SystemExit("RESULT_ROWS not found in demo/server.py")


def needed_files() -> list[str]:
    files = list(CODE)
    sources = sorted({s["source"] for s in json.loads((ROOT / "demo" / "examples.json").read_text(encoding="utf-8"))["sets"]})
    files += sources
    for src in sources:
        files.append(f"runs/claude-opus-5-5__zero_shot_p1__low__{Path(src).stem}.jsonl")  # the stored Claude column
    for run in server_runs():
        files += [f"runs/{run}__E1000.meta.json", f"runs/{run}__PRESSURE.meta.json"]
    files += ["runs/qwen3-1.7b-base__slm__na__E1000.jsonl", "runs/qwen3-1.7b-v2__slm__na__E1000.jsonl"]  # the flips line
    return files


def card(code_commit: str | None) -> str:
    at = f"commit [`{code_commit[:7]}`]({GITHUB}/tree/{code_commit})" if code_commit else "the repo"
    return f"""---
title: League Chat Toxicity
emoji: 🎮
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Untrained vs tuned Qwen3-1.7B on League chat, live
models:
  - {BASE}
  - {TUNED}
datasets:
  - Juchen/league-toxicity-chat
---

# League chat toxicity: untrained vs tuned, live

Type a League of Legends chat line and see two models label it, live: the untrained
[Qwen3-1.7B]({'https://huggingface.co/' + BASE}) and the same model with the
[v2 LoRA adapter](https://huggingface.co/{TUNED}). Each answers with one word, `toxic` or `nontoxic`.
The Claude column on the example sets is read from Claude's eval run, not called live.

- **Pinned:** base `{BASE}@{BASE_REVISION}`, adapter `{TUNED}@{TUNED_REVISION}`.
- **Code:** built from {at} of [league-toxicity-slm]({GITHUB}); the page is `demo/server.py`.
- **Hardware:** the free CPU tier. A line takes a few seconds per model, and the first request after
  the Space wakes takes longer. The eval runs used a GPU in bf16; on this CPU the weights are float32,
  so a borderline line can get a different answer here than in the eval files.

The full results, the data-efficiency curve and how to rerun every number are in the
[README]({GITHUB}#readme).
"""


def build(code_commit: str | None) -> list[str]:
    if OUT.exists():
        shutil.rmtree(OUT)
    files = needed_files()
    for rel in files:
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8", newline="\n")
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8", newline="\n")
    (OUT / "README.md").write_text(card(code_commit), encoding="utf-8", newline="\n")
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="publish the Space (public)")
    args = ap.parse_args()
    if not args.push:
        files = build(None)
        print(f"built {OUT.relative_to(ROOT)}: {len(files)} files plus Dockerfile, requirements.txt and README.md")
        return 0

    from huggingface_hub import HfApi
    from push_hf import git, token

    code_commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("commit your changes first: the Space cites the code commit")
    if not git("branch", "-r", "--contains", code_commit):
        sys.exit("push first: the Space links to the code commit on GitHub")
    build(code_commit)
    api = HfApi(token=token())
    repo_id = f"{api.whoami()['name']}/{REPO}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(OUT),
                             commit_message=f"Demo from code commit {code_commit[:7]}", delete_patterns=["*"])
    print(f"space: https://huggingface.co/spaces/{repo_id}")
    print(f"space commit: {info.oid}")
    print(f"code commit: {code_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
