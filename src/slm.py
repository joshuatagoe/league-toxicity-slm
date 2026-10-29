"""Load the small model (base or base + LoRA adapter) and label messages with it.

Runs in the training environment (.venv-train), which has torch and transformers.
The prompt is rendered exactly as in training: slm_prompt.messages() through the
model's own chat template, with thinking off.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from slm_prompt import messages


def quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def render(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load(model_ref: str, revision: str | None = None, four_bit: bool = False):
    """model_ref is a full model (hub id or folder) or a LoRA adapter folder/repo.

    Evaluation loads the base in bf16 by default: in 4-bit, the untrained Qwen3-1.7B
    often echoes a word from the message instead of answering, which made the base
    score meaningless. Training still uses the 4-bit base (QLoRA).
    """
    adapter_cfg = None
    local = Path(model_ref)
    if (local / "adapter_config.json").exists():
        adapter_cfg = json.loads((local / "adapter_config.json").read_text())
    else:
        try:
            from huggingface_hub import hf_hub_download

            adapter_cfg = json.loads(Path(hf_hub_download(model_ref, "adapter_config.json", revision=revision)).read_text())
        except Exception:
            adapter_cfg = None
    base = adapter_cfg["base_model_name_or_path"] if adapter_cfg else model_ref
    tokenizer = AutoTokenizer.from_pretrained(base, revision=None if adapter_cfg else revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base,
        revision=None if adapter_cfg else revision,
        quantization_config=quant_config() if four_bit else None,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if adapter_cfg:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_ref, revision=revision)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def label(tokenizer, model, texts: list[str], batch_size: int = 32, max_new_tokens: int = 6) -> list[str]:
    out = []
    for i in range(0, len(texts), batch_size):
        prompts = [render(tokenizer, t) for t in texts[i : i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        out += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return [o.strip() for o in out]
