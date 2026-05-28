"""Typed configuration loader.

We use Pydantic so that:

- Bad config files fail loudly with a single, readable validation error
  instead of obscure ``KeyError`` deep inside the trainer.
- Defaults live in one place (the model definition), not scattered across
  argparse calls and shell scripts.
- A run's effective config can be serialized verbatim to the experiment
  tracker for reproducibility.

Configs are YAML; CLI flags can override individual fields with dotted
paths (e.g. ``--override training.num_epochs=2``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    base_model: str = Field(..., description="Hugging Face model id or local path.")
    revision: str | None = Field(
        default=None, description="Optional model revision/SHA for reproducibility."
    )
    trust_remote_code: bool = True
    load_in_4bit: bool = Field(
        default=False,
        description="Use bitsandbytes 4-bit quantization (QLoRA). Requires CUDA.",
    )


class LoraConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    @field_validator("r", "alpha")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("r and alpha must be positive integers")
        return v


class DataConfig(BaseModel):
    train_file: Path
    val_file: Path
    max_seq_len: int = 1024


class TrainingConfig(BaseModel):
    output_dir: Path
    num_epochs: float = 2.0
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_total_limit: int = 1
    seed: int = 42


class TrackingConfig(BaseModel):
    enabled: bool = False
    project: str = "math-lora"
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Top-level config combining all sub-sections."""

    model: ModelConfig
    lora: LoraConfig = LoraConfig()
    data: DataConfig
    training: TrainingConfig
    tracking: TrackingConfig = TrackingConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def apply_overrides(self, overrides: list[str]) -> "RunConfig":
        """Apply ``section.key=value`` style overrides from the CLI.

        Values are parsed as YAML scalars so ``true``/``false`` and numbers
        round-trip correctly.
        """

        if not overrides:
            return self

        data: dict[str, Any] = self.model_dump()
        for entry in overrides:
            if "=" not in entry:
                raise ValueError(f"Bad override (expected key=value): {entry!r}")
            key, value = entry.split("=", 1)
            parsed = yaml.safe_load(value)
            target = data
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise KeyError(f"Override path not found: {key}")
                target = target[part]
            target[parts[-1]] = parsed

        return RunConfig.model_validate(data)
