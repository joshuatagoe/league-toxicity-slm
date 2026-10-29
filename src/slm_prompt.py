"""The one prompt the small model sees, at training and at evaluation alike.

Keeping it in one place means the tuned model never meets a format it wasn't
trained on.
"""

SYSTEM = (
    "You moderate League of Legends chat. Label the player's message toxic or nontoxic, "
    "the way experienced League players would. Reply with exactly one word: toxic or nontoxic."
)
LABELS = {1: "toxic", 0: "nontoxic"}


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
