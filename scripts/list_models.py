"""Print the model IDs this account can call, to pick the grid's models.

Usage: python -I scripts/list_models.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_ceiling import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import anthropic  # noqa: E402
import openai  # noqa: E402

try:
    ids = sorted(m.id for m in openai.OpenAI().models.list() if m.id.startswith(("gpt-5", "gpt-4o", "gpt-4.1")))
    print("OpenAI:", ", ".join(ids))
except openai.OpenAIError as e:
    print(f"OpenAI: {type(e).__name__}: {e}")

try:
    print("Anthropic:", ", ".join(m.id for m in anthropic.Anthropic().models.list()))
except anthropic.AnthropicError as e:
    print(f"Anthropic: {type(e).__name__}: {e}")
