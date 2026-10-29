"""QLoRA fine-tuning: 4-bit frozen base, small trainable LoRA adapters, loss on the label only.

Usage (training environment):
  .venv-train/Scripts/python train/train_qlora.py --base-model Qwen/Qwen3-1.7B --n 10919 --out models/qwen3-1.7b-v1

--n takes the first N rows of the (pre-shuffled) training file, so N, N/2, N/4, N/8
are nested subsets for the data-efficiency curve. Saves the adapter and train_meta.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from slm import quant_config, render  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-file", default="data/sft/train_v1.jsonl")
    ap.add_argument("--n", type=int, help="use the first N training rows (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke runs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    train_path = ROOT / args.train_file
    rows = [json.loads(l) for l in train_path.open(encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    eos = tokenizer.eos_token
    data = Dataset.from_list(
        [{"prompt": render(tokenizer, r["prompt"][1]["content"]), "completion": r["completion"][0]["content"] + eos} for r in rows]
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant_config(), dtype=torch.bfloat16, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")

    out = ROOT / args.out
    steps_per_epoch = -(-len(rows) // (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    config = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * total_steps)),  # transformers 5 has no warmup_ratio
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        seed=args.seed,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=lora)

    first = trainer.train_dataset[0]
    if "labels" in first:
        trained = [t for t in first["labels"] if t != -100]
    else:
        trained = [t for t, m in zip(first["input_ids"], first["completion_mask"]) if m]
    print("columns:", trainer.train_dataset.column_names)
    print("first example, full text:", repr(tokenizer.decode(first["input_ids"])[-120:]))
    print("first example, trained tokens:", repr(tokenizer.decode(trained)))

    start = time.time()
    result = trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "n": len(rows),
        "toxic": sum(r["label"] for r in rows),
        "args": vars(args),
        "train_loss": result.training_loss,
        "steps": result.global_step,
        "minutes": round((time.time() - start) / 60, 1),
        "git_commit": commit,
        "versions": {"torch": torch.__version__, "trl": __import__("trl").__version__, "peft": __import__("peft").__version__},
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n", "toxic", "train_loss", "steps", "minutes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
