"""Smoke tests for the project layout established in task 1.1.

These tests do not exercise any production logic; they only verify that the
package layout, version metadata, and component subpackages described in the
design document are importable. They will be superseded by the schema and
behavior tests added in subsequent tasks.
"""

from __future__ import annotations

import importlib

import pytest


COMPONENT_PACKAGES = [
    "math_lora",
    "math_lora.types",
    "math_lora.model_selector",
    "math_lora.planner",
    "math_lora.dataset_builder",
    "math_lora.lora_trainer",
    "math_lora.pipeline",
    "math_lora.evaluator",
    "math_lora.inference_server",
    "math_lora.tracker",
]


@pytest.mark.unit
@pytest.mark.parametrize("name", COMPONENT_PACKAGES)
def test_component_package_is_importable(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__name__ == name


@pytest.mark.unit
def test_top_level_version_is_exposed() -> None:
    import math_lora

    assert isinstance(math_lora.__version__, str)
    assert math_lora.__version__  # non-empty
