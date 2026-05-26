"""Hardware_Budget_Planner component (Requirement 2).

Loads Hardware_Profile and Budget_Profile, resolves the default Quantization_Mode,
projects wall-clock time, monetary cost, and peak VRAM, and emits a PreFlightReport
that halts the pipeline before any training step when projections exceed limits.

At end-of-run, when ``Hardware_Profile.deployment == "cloud"``, the planner also
emits a :class:`CostReconciliation` (task 3.9, Requirement 2.12). The cloud-vs-local
emission gating itself is wired by ``Training_Pipeline`` in task 12.3; this
package only delivers the pure arithmetic.
"""

from math_lora.planner.config_loader import (
    load_budget_profile,
    load_hardware_profile,
)
from math_lora.planner.cost_reconciliation import (
    CostReconciliation,
    PreFlightCostInputs,
    reconcile_cost,
)
from math_lora.planner.quantization_default import resolve_quantization_mode

__all__ = [
    "load_hardware_profile",
    "load_budget_profile",
    "resolve_quantization_mode",
    "CostReconciliation",
    "PreFlightCostInputs",
    "reconcile_cost",
]
