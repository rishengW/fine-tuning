"""Model_Selector component (Requirement 1).

Selects a feasible Base_Model from a list of candidates given a Hardware_Profile,
Quantization_Mode, and training sequence length, and emits a SelectionReport.

This sub-package is built up over tasks 2.1 -- 2.8 of the implementation plan.
The public surface presently exposes the VRAM-estimation primitives delivered
by task 2.1, which are also consumed by ``Hardware_Budget_Planner``
(Requirement 2.5) so that both components share a single source of truth for
the documented coefficients (Requirement 1.3).
"""

from math_lora.model_selector.vram import (
    VRAMCoefficients,
    VRAMEstimate,
    estimate_min_vram_gb,
)

__all__ = [
    "VRAMCoefficients",
    "VRAMEstimate",
    "estimate_min_vram_gb",
]
