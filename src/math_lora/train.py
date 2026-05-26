"""Minimal LoRA fine-tuning script.

Loads a base causal LM, applies a LoRA adapter via `peft`, trains on a JSONL
dataset of chat-style examples, and saves the adapter to disk.

Each line in the JSONL files is expected to look like:

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

Usage (from the project root, with the venv activated):

    python -m math_lora.train \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --train-file data/train.jsonl \
        --val-file data/val.jsonl \
        --output-dir outputs/adapter
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str
    train_file: Path
    val_file: Path
    output_dir: Path
    max_seq_len: int
    num_epochs: float
    per_device_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Minimal LoRA fine-tuning runner.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(records: list[dict[str, Any]], tokenizer, max_seq_len: int) -> Dataset:
    """Render chat messages with the tokenizer's chat template, then tokenize."""

    def render(example: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds = Dataset.from_list(records)
    ds = ds.map(render, remove_columns=ds.column_names)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()

    print(f"[math-lora] loading tokenizer + base model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print(f"[math-lora] applying LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[math-lora] loading data: train={cfg.train_file}, val={cfg.val_file}")
    train_records = load_jsonl(cfg.train_file)
    val_records = load_jsonl(cfg.val_file)
    train_ds = build_dataset(train_records, tokenizer, cfg.max_seq_len)
    val_ds = build_dataset(val_records, tokenizer, cfg.max_seq_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("[math-lora] starting training")
    trainer.train()

    print(f"[math-lora] saving adapter to {cfg.output_dir}")
    model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print("[math-lora] done")


if __name__ == "__main__":
    main()
