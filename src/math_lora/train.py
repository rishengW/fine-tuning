"""LoRA fine-tuning entry point.

Driven by a YAML config (see ``configs/*.yaml``) so that runs are
reproducible and trackable, with optional CLI overrides for one-off
tweaks. Supports plain LoRA (default) and 4-bit QLoRA (when
``model.load_in_4bit=true``).

Each line of the input JSONL files is expected to look like::

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

Usage::

    python -m math_lora.train --config configs/qwen-0.5b-lora.yaml
    python -m math_lora.train --config configs/qwen-0.5b-lora.yaml \
        --override training.num_epochs=1 --override lora.r=4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from math_lora.config import RunConfig
from math_lora.logging_utils import WandbTracker, get_logger

log = get_logger("math_lora.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for math reasoning.")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. `--override training.num_epochs=1`.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_dataset(records: list[dict[str, Any]], tokenizer, max_seq_len: int) -> Dataset:
    def render(example: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds = Dataset.from_list(records)
    return ds.map(render, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _load_model(cfg: RunConfig):
    log.info("loading tokenizer + base model: %s", cfg.model.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.revision:
        model_kwargs["revision"] = cfg.model.revision

    if cfg.model.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("load_in_4bit requires CUDA + bitsandbytes")
        from transformers import BitsAndBytesConfig

        log.info("using 4-bit nf4 quantization (QLoRA)")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, **model_kwargs)

    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    log.info(
        "applying LoRA: r=%d alpha=%d dropout=%.2f targets=%s",
        cfg.lora.r,
        cfg.lora.alpha,
        cfg.lora.dropout,
        cfg.lora.target_modules,
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Tracking glue: stream Trainer metrics into the WandbTracker.
# ---------------------------------------------------------------------------
class _WandbForwardCallback(TrainerCallback):
    def __init__(self, tracker: WandbTracker) -> None:
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if logs and self.tracker.enabled:
            self.tracker.log({k: v for k, v in logs.items() if isinstance(v, (int, float))},
                             step=state.global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = RunConfig.from_yaml(args.config).apply_overrides(args.override)
    log.info("effective config:\n%s", cfg.model_dump_json(indent=2))

    tracker = WandbTracker(
        enabled=cfg.tracking.enabled,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name,
        tags=cfg.tracking.tags,
        config=cfg.model_dump(),
    )

    tokenizer, model = _load_model(cfg)

    log.info("loading data: train=%s val=%s", cfg.data.train_file, cfg.data.val_file)
    train_records = _load_jsonl(cfg.data.train_file)
    val_records = _load_jsonl(cfg.data.val_file)
    train_ds = _build_dataset(train_records, tokenizer, cfg.data.max_seq_len)
    val_ds = _build_dataset(val_records, tokenizer, cfg.data.max_seq_len)
    log.info("train_examples=%d val_examples=%d", len(train_ds), len(val_ds))

    cfg.training.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(cfg.training.output_dir),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.training.save_total_limit,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg.training.seed,
    )

    callbacks = [_WandbForwardCallback(tracker)] if tracker.enabled else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )

    log.info("starting training")
    trainer.train()

    log.info("saving adapter to %s", cfg.training.output_dir)
    model.save_pretrained(str(cfg.training.output_dir))
    tokenizer.save_pretrained(str(cfg.training.output_dir))

    # Persist the resolved config alongside the adapter for reproducibility.
    (cfg.training.output_dir / "run_config.json").write_text(
        cfg.model_dump_json(indent=2), encoding="utf-8"
    )

    tracker.finish()
    log.info("done")


if __name__ == "__main__":
    main()
