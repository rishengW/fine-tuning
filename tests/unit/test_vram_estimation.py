"""Unit tests for the VRAM-estimation primitives delivered by task 2.1.

These are example-based tests that complement the property-based test in
task 2.2 (`tests/property/test_vram_estimation.py`, registered later in the
plan). Together they cover Requirement 1.3 from two angles:

* The property test asserts the universal monotonicity / determinism /
  coefficient-reconstruction laws on randomly generated inputs.
* This module asserts the laws on a small set of carefully chosen
  example points -- including the exact mode-ordering called out by the
  task prompt and the per-term breakdown shape -- so a regression in
  any one of those laws fails fast with a readable error.

Each test names the law it exercises in its docstring and cites the
requirement clause it validates. The tests deliberately do not depend on
the absolute numeric value of the estimate (the activation proxy is a
calibrated approximation); they only depend on the laws that the
implementation guarantees.
"""

from __future__ import annotations

import math

import pytest

from math_lora.model_selector import (
    VRAMCoefficients,
    VRAMEstimate,
    estimate_min_vram_gb,
)
from math_lora.types import (
    BaseModelCandidate,
    HardwareProfile,
    QuantizationMode,
)
from math_lora.types.quantization import BYTES_PER_PARAM


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _candidate(param_count_b: float = 7.0) -> BaseModelCandidate:
    """Build a valid :class:`BaseModelCandidate` with the given size.

    Every other field is held constant across calls so that the only
    variable in the resulting estimate is ``param_count_b``. This keeps
    monotonicity tests focused on a single input dimension.
    """

    return BaseModelCandidate.parse(
        {
            "model_id": f"test/Model-{param_count_b}B",
            "revision": "main",
            "family": "qwen",
            "param_count_b": param_count_b,
            "license_id": "Apache-2.0",
            "license_allows_finetuning": True,
            "license_allows_adapter_redistribution": True,
            "license_allows_commercial_use": True,
            "native_context_length_tokens": 4096,
            "tokenizer_family": "qwen",
            "baseline_gsm8k": 0.5,
            "baseline_math": 0.4,
        }
    )


def _hardware_profile(vram_per_gpu_gb: int = 24) -> HardwareProfile:
    """Build a valid :class:`HardwareProfile`.

    The estimate itself does not depend on the profile (the function only
    accepts it for interface symmetry with the design diagram), but every
    test still passes a real profile so the call site reads naturally.
    """

    return HardwareProfile.parse(
        {
            "gpu_model": "RTX 4090",
            "gpu_count": 1,
            "vram_per_gpu_gb": vram_per_gpu_gb,
            "system_ram_gb": 64,
            "disk_space_gb": 1024,
            "accelerator_family": "cuda",
            "deployment": "local",
        }
    )


# Canonical mode ordering called out by Property 1: the estimate must be
# monotone non-increasing (equality allowed) as the mode walks through this
# tuple. ``fp16`` and ``bf16`` share a ``bytes_per_param`` so the two
# adjacent estimates are equal; ``int8`` and ``nf4`` strictly decrease.
_MODE_ORDER: tuple[QuantizationMode, ...] = ("fp16", "bf16", "int8", "nf4")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """The function is pure: same inputs -> bit-identical output.

    Validates the determinism leg of Property 1 (Requirement 1.3).
    """

    @pytest.mark.unit
    def test_repeated_call_returns_identical_estimate(self) -> None:
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        a = estimate_min_vram_gb(candidate, hw, "bf16", 2048)
        b = estimate_min_vram_gb(candidate, hw, "bf16", 2048)

        # Bit-identical on every field: a "pure function, no global state"
        # guarantee per the task prompt and Property 1.
        assert a.estimated_min_vram_gb == b.estimated_min_vram_gb
        assert a.base_weights_gb == b.base_weights_gb
        assert a.lora_gradient_gb == b.lora_gradient_gb
        assert a.optimizer_state_gb == b.optimizer_state_gb
        assert a.activation_gb == b.activation_gb
        assert a.overhead_gb == b.overhead_gb

    @pytest.mark.unit
    def test_repeated_call_with_explicit_coefficients_is_deterministic(self) -> None:
        # Custom coefficients must also produce deterministic output --
        # the function does not cache or mutate its coefficient input.
        candidate = _candidate(1.5)
        hw = _hardware_profile()
        coeffs = VRAMCoefficients(
            optimizer_state_multiplier=4.0,
            activation_coefficient=1.0,
            overhead_bytes=0.0,
            lora_trainable_param_ratio=0.01,
        )

        a = estimate_min_vram_gb(candidate, hw, "nf4", 1024, coefficients=coeffs)
        b = estimate_min_vram_gb(candidate, hw, "nf4", 1024, coefficients=coeffs)

        assert a == b


# ---------------------------------------------------------------------------
# Per-term breakdown / coefficient reconstruction
# ---------------------------------------------------------------------------


class TestPerTermBreakdown:
    """Per-term fields sum exactly to ``estimated_min_vram_gb``.

    Validates the "coefficients exposed in the selection report" half of
    Requirement 1.3 -- a consumer can audit each contribution.
    """

    @pytest.mark.unit
    def test_estimate_returns_a_vram_estimate_instance(self) -> None:
        # The result is the documented dataclass, not a bare float -- so
        # downstream callers can record the coefficients in the
        # selection_report (Requirement 1.3).
        result = estimate_min_vram_gb(_candidate(), _hardware_profile(), "bf16", 2048)
        assert isinstance(result, VRAMEstimate)

    @pytest.mark.unit
    def test_per_term_breakdown_sums_to_total(self) -> None:
        result = estimate_min_vram_gb(_candidate(7.0), _hardware_profile(), "nf4", 2048)

        reconstructed = (
            result.base_weights_gb
            + result.lora_gradient_gb
            + result.optimizer_state_gb
            + result.activation_gb
            + result.overhead_gb
        )
        # Float addition is associative within a couple of ULPs at this
        # magnitude; ``isclose`` with a relative tolerance of 1e-12
        # captures that without being so loose that a real reconstruction
        # bug would slip through.
        assert math.isclose(
            reconstructed, result.estimated_min_vram_gb, rel_tol=1e-12, abs_tol=0.0
        )

    @pytest.mark.unit
    def test_estimate_exposes_all_required_coefficient_fields(self) -> None:
        # The task prompt enumerates the coefficient fields that must be
        # carried alongside the estimate. Asserting their presence here
        # protects the dataclass shape against accidental field removal.
        result = estimate_min_vram_gb(_candidate(), _hardware_profile(), "bf16", 2048)
        coeffs = result.coefficients

        # ``bytes_per_param`` carries one entry per QuantizationMode.
        assert set(coeffs.bytes_per_param.keys()) == set(BYTES_PER_PARAM.keys())
        for mode, expected in BYTES_PER_PARAM.items():
            assert coeffs.bytes_per_param[mode] == expected

        # The four scalar coefficients enumerated by task 2.1.
        assert coeffs.optimizer_state_multiplier > 0
        assert coeffs.activation_coefficient >= 0
        assert coeffs.overhead_bytes >= 0
        # LoRA on q_proj/v_proj at r=16 ~ 0.5% of total params per the
        # QLoRA paper; the default sits in (0, 0.05].
        assert 0.0 < coeffs.lora_trainable_param_ratio < 0.05

    @pytest.mark.unit
    def test_overhead_term_equals_overhead_bytes_in_gb(self) -> None:
        # The overhead term is independent of every other input -- it is
        # *the* coefficient. This test fixes that exact invariant so a
        # future refactor that accidentally folds overhead into another
        # term fails immediately.
        coeffs = VRAMCoefficients(overhead_bytes=2.0 * (1 << 30))
        result = estimate_min_vram_gb(
            _candidate(7.0), _hardware_profile(), "bf16", 2048, coefficients=coeffs
        )
        assert math.isclose(result.overhead_gb, 2.0, rel_tol=1e-12, abs_tol=0.0)


# ---------------------------------------------------------------------------
# Monotonicity: sequence_length
# ---------------------------------------------------------------------------


class TestSequenceLengthMonotonicity:
    """Estimate is non-decreasing in ``sequence_length``.

    Validates the seq-length leg of Property 1 (Requirement 1.3).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", _MODE_ORDER)
    def test_longer_sequence_does_not_decrease_estimate(
        self, mode: QuantizationMode
    ) -> None:
        # Holding everything else constant, doubling the sequence length
        # cannot reduce the estimate. We test across all four modes so a
        # mode-specific regression in the activation term is caught.
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        small = estimate_min_vram_gb(candidate, hw, mode, 128)
        big = estimate_min_vram_gb(candidate, hw, mode, 8192)

        assert big.estimated_min_vram_gb >= small.estimated_min_vram_gb

    @pytest.mark.unit
    def test_longer_sequence_strictly_increases_activation_term(self) -> None:
        # Stronger fact: the activation term itself is strictly increasing
        # in seq_len (the linear coefficient is positive). This guards
        # against a subtle bug where the seq term saturates.
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        a = estimate_min_vram_gb(candidate, hw, "bf16", 1024)
        b = estimate_min_vram_gb(candidate, hw, "bf16", 4096)

        assert b.activation_gb > a.activation_gb


# ---------------------------------------------------------------------------
# Monotonicity: param_count_b
# ---------------------------------------------------------------------------


class TestParamCountMonotonicity:
    """Estimate is non-decreasing in ``param_count_b``.

    Validates the parameter-count leg of Property 1 (Requirement 1.3).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", _MODE_ORDER)
    def test_more_params_does_not_decrease_estimate(
        self, mode: QuantizationMode
    ) -> None:
        hw = _hardware_profile()

        small = estimate_min_vram_gb(_candidate(1.5), hw, mode, 2048)
        big = estimate_min_vram_gb(_candidate(7.0), hw, mode, 2048)

        assert big.estimated_min_vram_gb >= small.estimated_min_vram_gb

    @pytest.mark.unit
    def test_more_params_strictly_increases_base_weights_term(self) -> None:
        # The base-weights term is strictly increasing in param count for
        # any mode whose ``bytes_per_param`` is positive (all four are).
        hw = _hardware_profile()

        a = estimate_min_vram_gb(_candidate(1.5), hw, "bf16", 2048)
        b = estimate_min_vram_gb(_candidate(7.0), hw, "bf16", 2048)

        assert b.base_weights_gb > a.base_weights_gb


# ---------------------------------------------------------------------------
# Monotonicity: quantization mode ordering
# ---------------------------------------------------------------------------


class TestModeMonotonicity:
    """Estimate is non-increasing across ``[fp16, bf16, int8, nf4]``.

    Validates the mode-ordering leg of Property 1 (Requirement 1.3). The
    ordering is fixed by the design's Research Notes ("the published
    memory-cost ordering ``fp16 (2 B) -> bf16 (2 B) -> int8 (1 B) -> nf4
    (0.5 B)``"), and equality at the ``fp16 -> bf16`` step is acceptable
    because both modes share ``bytes_per_param == 2.0``.
    """

    @pytest.mark.unit
    def test_estimate_is_non_increasing_along_mode_order(self) -> None:
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        estimates = [
            estimate_min_vram_gb(candidate, hw, mode, 2048).estimated_min_vram_gb
            for mode in _MODE_ORDER
        ]

        # Each adjacent pair must satisfy ``prev >= next``. We use a list
        # comprehension so the failing pair is named in the assertion
        # message rather than being hidden inside ``all(...)``.
        for prev_mode, next_mode, prev, nxt in zip(
            _MODE_ORDER, _MODE_ORDER[1:], estimates, estimates[1:]
        ):
            assert prev >= nxt, (
                f"VRAM estimate must not increase from {prev_mode} -> "
                f"{next_mode}: got {prev} -> {nxt}"
            )

    @pytest.mark.unit
    def test_fp16_and_bf16_are_equal(self) -> None:
        # Both modes use 2 bytes per parameter, so every term -- including
        # the total -- is bit-identical. This is the equality case that
        # task 2.1 explicitly calls out.
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        fp16 = estimate_min_vram_gb(candidate, hw, "fp16", 2048)
        bf16 = estimate_min_vram_gb(candidate, hw, "bf16", 2048)

        assert fp16.estimated_min_vram_gb == bf16.estimated_min_vram_gb
        assert fp16.base_weights_gb == bf16.base_weights_gb

    @pytest.mark.unit
    def test_nf4_strictly_smaller_than_bf16(self) -> None:
        # bf16 = 2.0 B/param vs nf4 = 0.5 B/param: the base-weights term
        # quarters, so the total must strictly decrease.
        candidate = _candidate(7.0)
        hw = _hardware_profile()

        bf16 = estimate_min_vram_gb(candidate, hw, "bf16", 2048)
        nf4 = estimate_min_vram_gb(candidate, hw, "nf4", 2048)

        assert nf4.estimated_min_vram_gb < bf16.estimated_min_vram_gb
