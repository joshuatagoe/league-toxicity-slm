"""Push a trained LoRA adapter to the Hugging Face Hub as a public model repo.

Usage (training environment, from the repo root, after committing and pushing):
  .venv-train/Scripts/python -I scripts/push_hf.py --model models/qwen3-1.7b-v2 --repo league-toxicity-qwen3-1.7b

Reads HF_TOKEN from .env. Uploads the adapter, tokenizer, chat template and
train_meta.json, plus docs/MODEL_CARD.md as the repo's README.md with the repo id
and the code commit filled in. Leaves out checkpoints/ and training_args.bin.
Prints the repo id and the Hub commit hash to quote in the submission.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "train_meta.json"]


def token() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "HF_TOKEN" and value.strip():
                return value.strip().strip('"').strip("'")
    return os.environ["HF_TOKEN"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local adapter folder")
    ap.add_argument("--repo", required=True, help="repo name, without the user")
    args = ap.parse_args()

    # The card cites the code commit, so it must be committed and on GitHub.
    code_commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("commit your changes first: the model card cites the code commit")
    if not git("branch", "-r", "--contains", code_commit):
        sys.exit("push first: the model card links to the code commit on GitHub")

    api = HfApi(token=token())
    repo_id = f"{api.whoami()['name']}/{args.repo}"
    card = (ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    card = card.replace("{repo_id}", repo_id).replace("{code_commit}", code_commit)
    model = ROOT / args.model
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(model / f)) for f in FILES]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))

    api.create_repo(repo_id, private=False, exist_ok=True)
    info = api.create_commit(repo_id, operations=ops, commit_message=f"Upload {args.model} from code commit {code_commit[:7]}")
    print(f"repo: https://huggingface.co/{repo_id}")
    print(f"hub commit: {info.oid}")
    print(f"code commit: {code_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
