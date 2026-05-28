"""Unit tests for the YAML config loader and CLI override layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from math_lora.config import RunConfig

_BASIC_YAML = """
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
data:
  train_file: data/train.jsonl
  val_file: data/val.jsonl
training:
  output_dir: outputs/test
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_minimal_config(tmp_path: Path) -> None:
    cfg = RunConfig.from_yaml(_write(tmp_path, _BASIC_YAML))
    assert cfg.model.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.r == 8  # default
    assert cfg.training.output_dir == Path("outputs/test")


def test_invalid_lora_rank_rejected(tmp_path: Path) -> None:
    bad = _BASIC_YAML + "\nlora:\n  r: 0\n"
    with pytest.raises(Exception):
        RunConfig.from_yaml(_write(tmp_path, bad))


def test_overrides_apply(tmp_path: Path) -> None:
    cfg = RunConfig.from_yaml(_write(tmp_path, _BASIC_YAML))
    new = cfg.apply_overrides(
        ["training.num_epochs=1", "lora.r=16", "tracking.enabled=true"]
    )
    assert new.training.num_epochs == 1
    assert new.lora.r == 16
    assert new.tracking.enabled is True
    # Original unchanged
    assert cfg.training.num_epochs == 2.0


def test_override_bad_key_rejected(tmp_path: Path) -> None:
    cfg = RunConfig.from_yaml(_write(tmp_path, _BASIC_YAML))
    with pytest.raises(KeyError):
        cfg.apply_overrides(["does_not_exist.x=1"])


def test_override_missing_equals_rejected(tmp_path: Path) -> None:
    cfg = RunConfig.from_yaml(_write(tmp_path, _BASIC_YAML))
    with pytest.raises(ValueError):
        cfg.apply_overrides(["training.num_epochs"])
