"""Property-based tests for ``estimate_min_vram_gb`` (task 2.2).

This module implements **Property 1** from the design document:

    *For any* candidate base model, hardware profile, sequence length, and
    quantization mode, the VRAM estimation function SHALL be deterministic
    (same inputs produce the same estimate), monotonically non-decreasing
    in ``param_count_b`` and ``sequence_length``, and monotonically
    non-increasing as ``Quantization_Mode`` moves from ``fp16`` to ``bf16``
    to ``int8`` to ``nf4`` (all other inputs held constant). The
    coefficients exposed in the selection report SHALL exactly explain the
    computed estimate.

**Validates: Requirements 1.3**

The four legs of Property 1 are encoded as four separate ``@given``-decorated
test functions so a counterexample names exactly the law that broke:

* :func:`test_estimate_is_deterministic` -- determinism leg
* :func:`test_estimate_is_monotone_non_decreasing_in_param_count_b`
* :func:`test_estimate_is_monotone_non_decreasing_in_sequence_length`
* :func:`test_estimate_is_monotone_non_increasing_across_mode_order`
* :func:`test_exposed_coefficients_reconstruct_estimate_exactly`

The Hypothesis profile is loaded by :mod:`tests.conftest` and guarantees
``max_examples >= 100`` per the design rule, so this file does not override
``settings`` itself.

Generators
----------

We reuse :func:`tests.property.strategies.valid_base_model_candidates` and
:func:`tests.property.strategies.valid_hardware_profiles` so every
generated input parses cleanly through the pydantic schemas; that keeps the
search focused on the input space the function actually accepts at runtime
and avoids spending budget on values that the schema would reject before
the function ever runs.

For ``sequence_length`` we draw integers in ``[128, 8192]`` -- the closed
interval declared in Requirement 1.2 -- and for ``Quantization_Mode`` we
sample from :data:`math_lora.types.quantization.QUANTIZATION_MODES`.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from math_lora.model_selector import estimate_min_vram_gb
from math_lora.types import BaseModelCandidate, HardwareProfile
from math_lora.types.quantization import (
    BYTES_PER_PARAM,
    QUANTIZATION_MODES,
    QuantizationMode,
)

from tests.property.strategies import (
    valid_base_model_candidates,
    valid_hardware_profiles,
)


# ---------------------------------------------------------------------------
# Local strategies for the inputs that aren't already in strategies.py
# ---------------------------------------------------------------------------


# Sequence length range from Requirement 1.2: closed interval ``[128, 8192]``.
# We mix the documented endpoints with arbitrary values so Hypothesis hits
# the boundaries during shrink while still exploring the interior.
_sequence_lengths: st.SearchStrategy[int] = st.one_of(
    st.sampled_from((128, 256, 512, 1024, 2048, 4096, 8192)),
    st.integers(min_value=128, max_value=8192),
)

#: Strategy producing one of the four documented quantization modes.
_quantization_modes: st.SearchStrategy[QuantizationMode] = st.sampled_from(
    QUANTIZATION_MODES
)

#: Canonical ordered tuple of modes from highest- to lowest-cost. Matches
#: the order called out by Property 1 in the design and by the task prompt.
_MODE_ORDER: tuple[QuantizationMode, ...] = ("fp16", "bf16", "int8", "nf4")


def _replace_param_count(
    candidate: BaseModelCandidate, new_param_count_b: float
) -> BaseModelCandidate:
    """Return a copy of ``candidate`` with ``param_count_b`` overwritten.

    Used by the param-count monotonicity test to hold every other field
    constant while sweeping the parameter count. We round-trip through
    ``model_dump`` + ``parse`` so the new value is re-validated by the
    pydantic schema (``param_count_b > 0``); the same code path that real
    callers use, so any schema regression surfaces here too.
    """

    payload = candidate.model_dump()
    payload["param_count_b"] = new_param_count_b
    return BaseModelCandidate.parse(payload)


# ---------------------------------------------------------------------------
# Property 1 leg A: determinism
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    candidate=valid_base_model_candidates(),
    hw=valid_hardware_profiles(),
    mode=_quantization_modes,
    seq_len=_sequence_lengths,
)
def test_estimate_is_deterministic(
    candidate: BaseModelCandidate,
    hw: HardwareProfile,
    mode: QuantizationMode,
    seq_len: int,
) -> None:
    """Repeated invocation with identical inputs returns identical output.

    Validates the determinism leg of Property 1 (Requirement 1.3). The
    function is documented as a pure function with no global state, so
    every field of the returned :class:`VRAMEstimate` -- including the
    ``coefficients`` mapping -- must be bit-for-bit equal across calls.
    """

    a = estimate_min_vram_gb(candidate, hw, mode, seq_len)
    b = estimate_min_vram_gb(candidate, hw, mode, seq_len)

    # Field-by-field bit-equality. ``==`` on the dataclass would also work
    # here (frozen dataclasses compare by field), but spelling it out
    # gives a more readable failure message when one specific term is the
    # culprit.
    assert a.estimated_min_vram_gb == b.estimated_min_vram_gb
    assert a.base_weights_gb == b.base_weights_gb
    assert a.lora_gradient_gb == b.lora_gradient_gb
    assert a.optimizer_state_gb == b.optimizer_state_gb
    assert a.activation_gb == b.activation_gb
    assert a.overhead_gb == b.overhead_gb
    assert a.coefficients == b.coefficients


# ---------------------------------------------------------------------------
# Property 1 leg B: monotone non-decreasing in param_count_b
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    candidate=valid_base_model_candidates(),
    hw=valid_hardware_profiles(),
    mode=_quantization_modes,
    seq_len=_sequence_lengths,
    delta=st.floats(
        min_value=1e-6,
        max_value=60.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_estimate_is_monotone_non_decreasing_in_param_count_b(
    candidate: BaseModelCandidate,
    hw: HardwareProfile,
    mode: QuantizationMode,
    seq_len: int,
    delta: float,
) -> None:
    """Increasing ``param_count_b`` cannot decrease the estimate.

    Validates the parameter-count leg of Property 1 (Requirement 1.3).
    Holding every other input fixed, the estimate is a sum of terms each
    of which is non-decreasing in ``param_count_b``: ``base_weights`` is
    linear with a non-negative coefficient (``bytes_per_param`` is
    non-negative for every mode), the LoRA gradient and optimizer terms
    are linear in the trainable count which is itself linear in
    ``param_count_b``, and the activation proxy is ``c * seq_len *
    sqrt(param_count_b * 1e9)`` (``sqrt`` is strictly increasing on the
    positive reals).
    """

    smaller = candidate
    bigger = _replace_param_count(candidate, candidate.param_count_b + delta)

    small_estimate = estimate_min_vram_gb(smaller, hw, mode, seq_len)
    big_estimate = estimate_min_vram_gb(bigger, hw, mode, seq_len)

    assert big_estimate.estimated_min_vram_gb >= small_estimate.estimated_min_vram_gb


# ---------------------------------------------------------------------------
# Property 1 leg C: monotone non-decreasing in sequence_length
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    candidate=valid_base_model_candidates(),
    hw=valid_hardware_profiles(),
    mode=_quantization_modes,
    seq_lens=st.tuples(_sequence_lengths, _sequence_lengths),
)
def test_estimate_is_monotone_non_decreasing_in_sequence_length(
    candidate: BaseModelCandidate,
    hw: HardwareProfile,
    mode: QuantizationMode,
    seq_lens: tuple[int, int],
) -> None:
    """Increasing ``sequence_length`` cannot decrease the estimate.

    Validates the sequence-length leg of Property 1 (Requirement 1.3).
    Only the activation term depends on ``sequence_length``, and that
    dependence is linear with a non-negative coefficient.
    """

    a, b = seq_lens
    short, long = (a, b) if a <= b else (b, a)

    short_estimate = estimate_min_vram_gb(candidate, hw, mode, short)
    long_estimate = estimate_min_vram_gb(candidate, hw, mode, long)

    assert long_estimate.estimated_min_vram_gb >= short_estimate.estimated_min_vram_gb


# ---------------------------------------------------------------------------
# Property 1 leg D: monotone non-increasing across mode order
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    candidate=valid_base_model_candidates(),
    hw=valid_hardware_profiles(),
    seq_len=_sequence_lengths,
)
def test_estimate_is_monotone_non_increasing_across_mode_order(
    candidate: BaseModelCandidate,
    hw: HardwareProfile,
    seq_len: int,
) -> None:
    """Estimate is non-increasing along ``[fp16, bf16, int8, nf4]``.

    Validates the mode-ordering leg of Property 1 (Requirement 1.3). The
    underlying reason is that ``BYTES_PER_PARAM`` is non-increasing along
    that ordering (``fp16 == bf16 == 2.0``, ``int8 == 1.0``,
    ``nf4 == 0.5``), and the only term that depends on the mode is
    ``base_weights = param_count * bytes_per_param``. Equality is allowed
    at the ``fp16 -> bf16`` step because both modes share the same
    ``bytes_per_param``.

    We compare every adjacent pair in the mode tuple in a single test, so
    a counterexample shrinks to the simplest set of ``(candidate, hw,
    seq_len)`` inputs that violate the law and the failing pair is named
    in the assertion message.
    """

    estimates = [
        estimate_min_vram_gb(candidate, hw, mode, seq_len).estimated_min_vram_gb
        for mode in _MODE_ORDER
    ]

    for prev_mode, next_mode, prev, nxt in zip(
        _MODE_ORDER, _MODE_ORDER[1:], estimates, estimates[1:]
    ):
        assert prev >= nxt, (
            f"VRAM estimate must not increase from {prev_mode} -> "
            f"{next_mode}: got {prev} -> {nxt}"
        )


# ---------------------------------------------------------------------------
# Property 1 leg E: exposed coefficients reconstruct the estimate exactly
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    candidate=valid_base_model_candidates(),
    hw=valid_hardware_profiles(),
    mode=_quantization_modes,
    seq_len=_sequence_lengths,
)
def test_exposed_coefficients_reconstruct_estimate_exactly(
    candidate: BaseModelCandidate,
    hw: HardwareProfile,
    mode: QuantizationMode,
    seq_len: int,
) -> None:
    """The exposed coefficients reconstruct the per-term breakdown exactly.

    Validates the "coefficients exposed in the selection report" half of
    Requirement 1.3. The :class:`VRAMEstimate` carries both the per-term
    breakdown (``base_weights_gb``, ``lora_gradient_gb``, ...) *and* the
    :class:`VRAMCoefficients` instance used to compute it, so a downstream
    consumer of the ``selection_report`` can multiply the recorded
    coefficients with the recorded inputs and recover the same numbers.

    Two checks:

    1. ``base_weights_gb + lora_gradient_gb + optimizer_state_gb +
       activation_gb + overhead_gb`` must equal ``estimated_min_vram_gb``
       within IEEE-754 rounding -- the per-term breakdown is the actual
       summands of the total.
    2. Recomputing each term from the exposed coefficients with the
       documented formula must reproduce the per-term breakdown exactly.
       This is the leg the task prompt explicitly calls out: *"verify
       exposed coefficients reconstruct the estimate exactly"*.
    """

    estimate = estimate_min_vram_gb(candidate, hw, mode, seq_len)
    coeffs = estimate.coefficients

    # ---- check 1: per-term breakdown sums to the total -----------------
    reconstructed_total = (
        estimate.base_weights_gb
        + estimate.lora_gradient_gb
        + estimate.optimizer_state_gb
        + estimate.activation_gb
        + estimate.overhead_gb
    )
    assert math.isclose(
        reconstructed_total,
        estimate.estimated_min_vram_gb,
        rel_tol=1e-12,
        abs_tol=0.0,
    ), (
        f"per-term breakdown {reconstructed_total} must sum to "
        f"estimated_min_vram_gb {estimate.estimated_min_vram_gb}"
    )

    # ---- check 2: exposed coefficients reproduce each term -------------
    # Constants that mirror the formula in src/math_lora/model_selector/vram.py.
    # We re-derive them here from public inputs (BYTES_PER_PARAM, the
    # design's "8 bytes per trainable param for AdamW") rather than from
    # private ``_PARAMS_PER_B`` / ``_BYTES_PER_GB`` constants, so this
    # test is a true second source for the formula -- a regression in the
    # implementation that quietly changes one of those constants will
    # surface here.
    params_per_b = 1.0e9
    bytes_per_gb = float(1 << 30)
    lora_gradient_bytes_per_param = 2.0  # bf16/fp16 LoRA weights, Req 4.8

    # Cross-check: the bytes-per-param entry the function used must match
    # the public table, so a future change to BYTES_PER_PARAM is reflected
    # in the recorded coefficients (Requirement 1.3 -- coefficients are
    # exposed verbatim).
    assert coeffs.bytes_per_param[mode] == BYTES_PER_PARAM[mode]

    trainable_params = (
        candidate.param_count_b * params_per_b * coeffs.lora_trainable_param_ratio
    )

    expected_base_weights_gb = (
        candidate.param_count_b * params_per_b * coeffs.bytes_per_param[mode]
    ) / bytes_per_gb
    expected_lora_gradient_gb = (
        trainable_params * lora_gradient_bytes_per_param
    ) / bytes_per_gb
    expected_optimizer_state_gb = (
        trainable_params * coeffs.optimizer_state_multiplier
    ) / bytes_per_gb
    expected_activation_gb = (
        coeffs.activation_coefficient
        * float(seq_len)
        * math.sqrt(candidate.param_count_b * params_per_b)
    ) / bytes_per_gb
    expected_overhead_gb = coeffs.overhead_bytes / bytes_per_gb

    # Each term must reconstruct exactly. We use a tight relative tolerance
    # rather than ``==`` so the test does not flake on the last bit of
    # IEEE-754 rounding when the function and this test reorder
    # multiplications differently (e.g. ``a * b / c`` vs ``a / c * b``).
    assert math.isclose(
        estimate.base_weights_gb, expected_base_weights_gb,
        rel_tol=1e-12, abs_tol=0.0,
    ), (
        f"base_weights_gb mismatch: estimate={estimate.base_weights_gb}, "
        f"reconstructed={expected_base_weights_gb}"
    )
    assert math.isclose(
        estimate.lora_gradient_gb, expected_lora_gradient_gb,
        rel_tol=1e-12, abs_tol=0.0,
    ), (
        f"lora_gradient_gb mismatch: estimate={estimate.lora_gradient_gb}, "
        f"reconstructed={expected_lora_gradient_gb}"
    )
    assert math.isclose(
        estimate.optimizer_state_gb, expected_optimizer_state_gb,
        rel_tol=1e-12, abs_tol=0.0,
    ), (
        f"optimizer_state_gb mismatch: estimate={estimate.optimizer_state_gb}, "
        f"reconstructed={expected_optimizer_state_gb}"
    )
    assert math.isclose(
        estimate.activation_gb, expected_activation_gb,
        rel_tol=1e-12, abs_tol=0.0,
    ), (
        f"activation_gb mismatch: estimate={estimate.activation_gb}, "
        f"reconstructed={expected_activation_gb}"
    )
    assert math.isclose(
        estimate.overhead_gb, expected_overhead_gb,
        rel_tol=1e-12, abs_tol=0.0,
    ), (
        f"overhead_gb mismatch: estimate={estimate.overhead_gb}, "
        f"reconstructed={expected_overhead_gb}"
    )
