"""One classify() call for every model family, plus two local baselines.

Model names: "claude-*" go to Anthropic, "baseline:*" run locally, anything else goes
to OpenAI. Credentials come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""
import re
import time
from dataclasses import dataclass

MAX_OUTPUT_TOKENS = 4096  # room for thinking tokens; a short label never needs this much
# Older OpenAI chat models take temperature 0, as in the paper; reasoning models take an effort.
OPENAI_TEMPERATURE_MODELS = ("gpt-4o", "gpt-4.1")

KEYWORDS = re.compile(
    r"\b(noob\w*|idiot\w*|retard\w*|tard|stupid|dumb|shut ?up|stfu|f+u+c*k\w*|fck\w*|shit\w*|"
    r"trash|feed\w*|troll\w*|report\w*|kys|cancer|bitch\w*|ass(hole)?)\b",
    re.I,
)


@dataclass
class Reply:
    text: str
    status: str  # ok | refusal | truncated | error
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


_clients = {}


def _anthropic():
    if "anthropic" not in _clients:
        import anthropic

        _clients["anthropic"] = anthropic.Anthropic(max_retries=5)
    return _clients["anthropic"]


def _openai():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(max_retries=5)
    return _clients["openai"]


def _call_anthropic(model: str, prompt: str, effort: str) -> Reply:
    import anthropic

    try:
        resp = _anthropic().messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        return Reply(text="", status="error", error=f"{type(e).__name__}: {e}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    status = {"refusal": "refusal", "max_tokens": "truncated"}.get(resp.stop_reason, "ok")
    return Reply(
        text=text,
        status=status,
        stop_reason=resp.stop_reason,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def _call_openai(model: str, prompt: str, effort: str) -> Reply:
    import openai

    kwargs = {"model": model, "input": prompt, "max_output_tokens": MAX_OUTPUT_TOKENS}
    if model.startswith(OPENAI_TEMPERATURE_MODELS):
        kwargs["temperature"] = 0
    else:
        kwargs["reasoning"] = {"effort": effort}
    try:
        resp = _openai().responses.create(**kwargs)
    except (openai.APIStatusError, openai.APIConnectionError) as e:
        return Reply(text="", status="error", error=f"{type(e).__name__}: {e}")
    status, text = "ok", resp.output_text or ""
    for item in resp.output:
        for part in getattr(item, "content", None) or []:
            if part.type == "refusal":
                status, text = "refusal", part.refusal
    if resp.status == "incomplete":
        status = "truncated"
    return Reply(
        text=text,
        status=status,
        stop_reason=resp.status,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def _call_baseline(model: str, message: str) -> Reply:
    if model == "baseline:always-nontoxic":
        return Reply(text="nontoxic", status="ok")
    if model == "baseline:keyword":
        return Reply(text="toxic" if KEYWORDS.search(message) else "nontoxic", status="ok")
    raise ValueError(f"unknown baseline {model}")


def classify(model: str, prompt: str, message: str, effort: str) -> Reply:
    start = time.perf_counter()
    if model.startswith("baseline:"):
        reply = _call_baseline(model, message)
    elif model.startswith("claude-"):
        reply = _call_anthropic(model, prompt, effort)
    else:
        reply = _call_openai(model, prompt, effort)
    reply.latency_ms = round((time.perf_counter() - start) * 1000)
    return reply
