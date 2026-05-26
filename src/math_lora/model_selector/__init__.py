"""Model_Selector component (Requirement 1).

Selects a feasible Base_Model from a list of candidates given a Hardware_Profile,
Quantization_Mode, and training sequence length, and emits a SelectionReport.

This sub-package is built up over tasks 2.1 -- 2.8 of the implementation plan:

* Task 2.1 (delivered): VRAM-estimation primitives
  (:func:`estimate_min_vram_gb`, :class:`VRAMEstimate`,
  :class:`VRAMCoefficients`).
* Task 2.3 (delivered): eligibility, feasibility, and scoring primitives
  (:func:`check_eligibility`, :func:`check_feasibility`,
  :func:`compute_score`, :func:`license_permissiveness`,
  :func:`pick_winner`, :func:`evaluate_candidate`) and the supporting
  dataclasses (:class:`EligibilityResult`, :class:`FeasibilityResult`,
  :class:`ScoringWeights`, :class:`ScoringInputs`, :class:`ScoredCandidate`,
  :class:`CandidateEntry`).
* Task 2.4 (next): the public ``select(...)`` orchestrator and the
  ``SelectionReport`` it emits. Will compose the primitives above.

The VRAM-estimation primitives are also consumed by
``Hardware_Budget_Planner`` (Requirement 2.5) so that both components share
a single source of truth for the documented coefficients (Requirement 1.3).
"""

from math_lora.model_selector.selection import (
    REQUIRED_FIELDS,
    TIE_BREAKER_RULE,
    CandidateEntry,
    EligibilityResult,
    FeasibilityResult,
    ScoredCandidate,
    ScoringInputs,
    ScoringWeights,
    check_eligibility,
    check_feasibility,
    compute_score,
    evaluate_candidate,
    license_permissiveness,
    pick_winner,
)
from math_lora.model_selector.vram import (
    VRAMCoefficients,
    VRAMEstimate,
    estimate_min_vram_gb,
)

__all__ = [
    # VRAM estimation (task 2.1)
    "VRAMCoefficients",
    "VRAMEstimate",
    "estimate_min_vram_gb",
    # Eligibility / feasibility / scoring (task 2.3)
    "REQUIRED_FIELDS",
    "TIE_BREAKER_RULE",
    "EligibilityResult",
    "FeasibilityResult",
    "ScoringWeights",
    "ScoringInputs",
    "ScoredCandidate",
    "CandidateEntry",
    "check_eligibility",
    "check_feasibility",
    "compute_score",
    "license_permissiveness",
    "pick_winner",
    "evaluate_candidate",
]
