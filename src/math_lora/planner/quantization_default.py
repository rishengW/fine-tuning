"""Default ``Quantization_Mode`` resolution for the ``Hardware_Budget_Planner``.

This module implements task 3.2 of the math-LoRA implementation plan, which
covers Requirement 2.4 of the feature spec:

    WHEN the Hardware_Profile declares VRAM per GPU below 24 gigabytes,
    THE Training_Pipeline SHALL set Quantization_Mode to ``nf4`` (QLoRA 4-bit)
    by default, and an operator SHALL be able to override this default by
    setting Quantization_Mode explicitly in the training configuration.

The threshold of 24 GB matches the QLoRA paper (Dettmers et al. 2023), which
demonstrates that 4-bit ``nf4`` quantization is the published reference
configuration for fitting 7B-class base models on a single consumer GPU.
At or above 24 GB, mixed-precision ``bf16`` is the standard high-VRAM
configuration consistent with Requirement 2.9 ("mixed precision in ``bf16``
or ``fp16``") and is the documented default this module returns when no
explicit override is supplied.

The resolution is intentionally a pure function (no I/O, no global state, no
randomness) so that it can be reused safely from the pre-flight gate and
from property tests without setup or teardown.
"""

from __future__ import annotations

from typing import Final

from math_lora.types import HardwareProfile, QuantizationMode


# ---------------------------------------------------------------------------
# Documented thresholds and defaults
# ---------------------------------------------------------------------------

#: VRAM threshold below which the planner defaults to ``nf4`` (Requirement 2.4).
#:
#: The comparison uses strict ``<`` so that exactly 24 GB hardware (e.g. a
#: single RTX 4090 or A10G) maps to the high-VRAM default, matching the spec
#: clause "VRAM per GPU below 24 gigabytes".
_NF4_VRAM_THRESHOLD_GB: Final[int] = 24

#: Default ``Quantization_Mode`` for hardware below the threshold.
#: Fixed by Requirement 2.4 to be ``nf4`` (the QLoRA 4-bit format).
_LOW_VRAM_DEFAULT: Final[QuantizationMode] = "nf4"

#: Default ``Quantization_Mode`` for hardware at or above the threshold.
#:
#: Requirement 2.4 only fixes the low-VRAM default; the high-VRAM default is
#: a documented design choice. We pick ``bf16`` because it is the standard
#: mixed-precision training mode on modern accelerators (Ampere and newer)
#: that already have at least 24 GB VRAM, and Requirement 2.9 explicitly
#: lists ``bf16`` as a supported VRAM-reduction toggle. Operators retain
#: full control: any explicit ``Quantization_Mode`` is honoured unchanged.
_HIGH_VRAM_DEFAULT: Final[QuantizationMode] = "bf16"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_quantization_mode(
    hardware_profile: HardwareProfile,
    explicit_mode: QuantizationMode | None,
) -> QuantizationMode:
    """Resolve the ``Quantization_Mode`` per Requirement 2.4.

    The function is total, pure, and deterministic: it has no side effects,
    performs no I/O, and depends only on its two arguments.

    Args:
        hardware_profile: The validated :class:`HardwareProfile` declaring
            ``vram_per_gpu_gb`` (an integer in ``[1, 1024]`` per
            Requirement 1.2 -- already enforced by the schema).
        explicit_mode: An operator-supplied override. When ``None`` the
            function picks the documented default for the hardware tier;
            otherwise it returns ``explicit_mode`` unchanged so the operator
            override always wins (Requirement 2.4: "an operator SHALL be
            able to override this default").

    Returns:
        The resolved ``Quantization_Mode``:

        * ``explicit_mode`` if it is not ``None`` (override respected);
        * ``"nf4"`` if ``hardware_profile.vram_per_gpu_gb < 24``
          (Requirement 2.4 default for sub-24 GB hardware);
        * ``"bf16"`` otherwise (documented default for >= 24 GB hardware,
          consistent with Requirement 2.9's mixed-precision toggles).

    Notes:
        Override precedence is intentional. Even on a sub-24 GB GPU an
        operator may legitimately want to train at ``int8`` (e.g. to compare
        memory/quality trade-offs) or at ``bf16`` for a small adapter, so
        the override is honoured before the VRAM-threshold check runs.
    """

    # Requirement 2.4 explicit-override clause: when the operator names a
    # mode in the training configuration we return it unchanged regardless
    # of the hardware tier. This is a deliberate early return so the
    # downstream branch never observes an explicit value.
    if explicit_mode is not None:
        return explicit_mode

    # Requirement 2.4 default clause for sub-24 GB hardware.
    if hardware_profile.vram_per_gpu_gb < _NF4_VRAM_THRESHOLD_GB:
        return _LOW_VRAM_DEFAULT

    # Documented default for >= 24 GB hardware. See module docstring for the
    # rationale; this branch covers the boundary case ``vram == 24`` because
    # the threshold uses strict ``<``.
    return _HIGH_VRAM_DEFAULT


__all__ = ["resolve_quantization_mode"]
