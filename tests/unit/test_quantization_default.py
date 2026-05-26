"""Unit tests for :func:`math_lora.planner.resolve_quantization_mode`.

Coverage
--------

Task 3.2 of the implementation plan covers Requirement 2.4 of the spec:

    WHEN the Hardware_Profile declares VRAM per GPU below 24 gigabytes,
    THE Training_Pipeline SHALL set Quantization_Mode to ``nf4`` (QLoRA
    4-bit) by default, and an operator SHALL be able to override this
    default by setting Quantization_Mode explicitly in the training
    configuration.

The function under test is a total, pure resolver. The tests below exercise
the three behavioural axes of that contract:

1. **Default selection for sub-24 GB hardware.** ``vram=8`` and
   ``vram=23`` must both resolve to ``nf4`` when no override is supplied.
2. **Default selection at and above the threshold.** ``vram=24`` (the
   documented boundary, where the strict ``<`` check evaluates ``False``)
   and ``vram=80`` (data-center class) must resolve to the documented
   high-VRAM default ``bf16``.
3. **Operator override is respected.** Any non-None ``explicit_mode`` is
   returned unchanged regardless of the hardware tier, including modes
   that are *not* the tier default (e.g. ``fp16`` on an 8 GB GPU, where
   an operator might be probing memory/quality trade-offs).
"""

from __future__ import annotations

from typing import Any

import pytest

from math_lora.planner import resolve_quantization_mode
from math_lora.types import HardwareProfile, QuantizationMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hw(vram_per_gpu_gb: int, **overrides: Any) -> HardwareProfile:
    """Build a :class:`HardwareProfile` with the given VRAM size.

    Only ``vram_per_gpu_gb`` matters for these tests, but the schema
    requires every field, so we provide consumer-grade defaults for
    everything else and let callers override by keyword if they ever
    need to (none of the cases below do).
    """

    payload: dict[str, Any] = {
        "gpu_model": "test-gpu",
        "gpu_count": 1,
        "vram_per_gpu_gb": vram_per_gpu_gb,
        "system_ram_gb": 32,
        "disk_space_gb": 512,
        "accelerator_family": "cuda",
        "deployment": "local",
    }
    payload.update(overrides)
    return HardwareProfile.parse(payload)


# ---------------------------------------------------------------------------
# Default resolution: no explicit override
# ---------------------------------------------------------------------------


class TestResolveQuantizationModeDefault:
    """Defaults selected when ``explicit_mode is None`` (Requirement 2.4)."""

    @pytest.mark.unit
    def test_vram_8_resolves_to_nf4(self) -> None:
        # 8 GB is the smallest VRAM tier the design considers (Req 2.11
        # consumer-GPU path); it sits well below the 24 GB threshold so
        # the planner must pick the QLoRA 4-bit default.
        assert resolve_quantization_mode(_hw(8), None) == "nf4"

    @pytest.mark.unit
    def test_vram_23_resolves_to_nf4(self) -> None:
        # 23 GB is one gigabyte below the 24 GB threshold and exercises
        # the boundary just *under* the cutoff -- the comparison uses
        # strict ``<`` so 23 must still resolve to ``nf4``.
        assert resolve_quantization_mode(_hw(23), None) == "nf4"

    @pytest.mark.unit
    def test_vram_24_resolves_to_bf16_boundary(self) -> None:
        # 24 GB is the spec's threshold value. Requirement 2.4 says
        # "below 24" maps to ``nf4`` -- i.e. exactly 24 is *not* below
        # 24, so the high-VRAM default applies. The documented choice
        # for >= 24 GB is ``bf16`` (see module docstring rationale).
        assert resolve_quantization_mode(_hw(24), None) == "bf16"

    @pytest.mark.unit
    def test_vram_80_resolves_to_bf16(self) -> None:
        # 80 GB is the A100-80GB / H100 tier; well above the threshold
        # and exercises the >= 24 GB branch with a generous margin.
        assert resolve_quantization_mode(_hw(80), None) == "bf16"


# ---------------------------------------------------------------------------
# Operator override: explicit_mode is honoured regardless of VRAM
# ---------------------------------------------------------------------------


class TestResolveQuantizationModeOverride:
    """Explicit overrides are returned unchanged (Requirement 2.4 clause 2)."""

    @pytest.mark.unit
    def test_vram_8_with_explicit_fp16_returns_fp16(self) -> None:
        # An operator on an 8 GB GPU might want fp16 to compare memory
        # behaviour against the nf4 default. The override must win, even
        # though fp16 on 8 GB is unlikely to fit a 7B model at training
        # time -- that's the planner's job to halt later (Requirement 2.6),
        # not the resolver's job to second-guess here.
        assert resolve_quantization_mode(_hw(8), "fp16") == "fp16"

    @pytest.mark.unit
    def test_vram_8_with_explicit_bf16_returns_bf16(self) -> None:
        # bf16 override on a sub-threshold GPU; the resolver must not
        # silently substitute nf4.
        assert resolve_quantization_mode(_hw(8), "bf16") == "bf16"

    @pytest.mark.unit
    def test_vram_24_with_explicit_nf4_returns_nf4(self) -> None:
        # Above-threshold hardware with an explicit nf4 override (e.g. an
        # operator running a much larger base model on a 24 GB card).
        assert resolve_quantization_mode(_hw(24), "nf4") == "nf4"

    @pytest.mark.unit
    def test_vram_24_with_explicit_int8_returns_int8(self) -> None:
        # Above-threshold hardware with int8 override; covers the fourth
        # member of the QuantizationMode literal so every documented
        # value appears at least once across the test suite.
        assert resolve_quantization_mode(_hw(24), "int8") == "int8"


# ---------------------------------------------------------------------------
# Property surface: determinism and totality across the documented modes
# ---------------------------------------------------------------------------


class TestResolveQuantizationModeContract:
    """Contract checks that complement the per-case defaults/overrides above."""

    @pytest.mark.unit
    def test_resolution_is_deterministic_for_repeated_calls(self) -> None:
        # The function is documented as pure; two calls with identical
        # inputs must return identical outputs without any hidden state.
        profile = _hw(8)
        first = resolve_quantization_mode(profile, None)
        second = resolve_quantization_mode(profile, None)
        assert first == second == "nf4"

    @pytest.mark.unit
    @pytest.mark.parametrize("explicit_mode", ["fp16", "bf16", "int8", "nf4"])
    def test_every_documented_mode_round_trips_as_override(
        self, explicit_mode: QuantizationMode
    ) -> None:
        # Sweeps the full QuantizationMode literal so a future addition
        # to the literal would force this test to be updated alongside
        # the resolver -- helpful as a regression tripwire.
        profile = _hw(16)
        assert resolve_quantization_mode(profile, explicit_mode) == explicit_mode
