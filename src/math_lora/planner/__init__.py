"""Hardware_Budget_Planner component (Requirement 2).

Loads Hardware_Profile and Budget_Profile, resolves the default Quantization_Mode,
projects wall-clock time, monetary cost, and peak VRAM, and emits a PreFlightReport
that halts the pipeline before any training step when projections exceed limits.
"""

from math_lora.planner.config_loader import (
    load_budget_profile,
    load_hardware_profile,
)

__all__ = [
    "load_hardware_profile",
    "load_budget_profile",
]
