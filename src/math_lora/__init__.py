"""math_lora: LoRA fine-tuning pipeline for mathematical reasoning.

The package is organized into one subpackage per component as described in the
design document at ``.kiro/specs/math-lora-finetuning/design.md``:

- :mod:`math_lora.types` - shared domain types and schemas
- :mod:`math_lora.model_selector` - Requirement 1
- :mod:`math_lora.planner` - Requirement 2 (Hardware_Budget_Planner)
- :mod:`math_lora.dataset_builder` - Requirement 3
- :mod:`math_lora.lora_trainer` - Requirement 4
- :mod:`math_lora.pipeline` - Requirement 5 (Training_Pipeline orchestrator)
- :mod:`math_lora.evaluator` - Requirement 6
- :mod:`math_lora.inference_server` - Requirement 7
- :mod:`math_lora.tracker` - Requirement 8 (Experiment_Tracker)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
