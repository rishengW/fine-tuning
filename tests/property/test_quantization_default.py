"""Property test for the default ``Quantization_Mode`` resolver.

Task 3.3 of the implementation plan asks for **Property 5** of the design
document, which validates Requirement 2.4:

    *For any* ``Hardware_Profile`` with ``vram_per_gpu_gb < 24`` and any
    training configuration in which ``quantization_mode`` is unset, the
    resolved ``Quantization_Mode`` SHALL be ``nf4``; *for any* training
    configuration in which ``quantization_mode`` is explicitly set, the
    resolved value SHALL equal the explicit setting.

The function under test is :func:`math_lora.planner.resolve_quantization_mode`,
implemented for task 3.2 in
``src/math_lora/planner/quantization_default.py``. It is a pure, total
resolver, so the property surface is small but worth pinning across the full
``[1, 1024]`` VRAM range and every member of the ``QuantizationMode`` literal.

The three checks below mirror the three behavioural axes called out in the
task prompt:

1. **Low-VRAM default.** For every ``HardwareProfile`` whose
   ``vram_per_gpu_gb`` is strictly below the 24 GB threshold, with no
   override, the resolver returns ``"nf4"``.
2. **Override is respected.** For every ``HardwareProfile`` and every
   ``QuantizationMode``, supplying that mode as the explicit override
   returns the same mode unchanged. This test deliberately covers VRAM
   above and below the threshold so any future change that silently
   substitutes the default would be caught.
3. **High-VRAM default mirrors the implementation.** For every
   ``HardwareProfile`` whose ``vram_per_gpu_gb`` is at or above the
   threshold, with no override, the resolver returns the documented
   high-VRAM default (``"bf16"``). Per the task prompt this property
   *mirrors* the implementation's documented default rather than
   assuming a value, so the constant is sourced from
   :mod:`math_lora.planner.quantization_default` to stay locked to the
   resolver itself.

All four members of the ``QuantizationMode`` literal are exercised across
the override property via the ``QUANTIZATION_MODES`` tuple.

Determinism note:
    The resolver is documented as pure and these tests do not seed any
    additional state. The Hypothesis profile registered in
    ``tests/conftest.py`` already enforces ``max_examples >= 100`` per the
    design rule, so no per-test ``settings`` override is needed.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from math_lora.planner import resolve_quantization_mode

# The high-VRAM default is sourced directly from the resolver module rather
# than re-asserted here. Per the task prompt for property 3 above, the
# property must "mirror the existing implementation" rather than assume a
# value. Importing the private constant keeps this test pinned to whatever
# the resolver currently documents and treats a future change to that
# default as a deliberate code change rather than a silent test mismatch.
from math_lora.planner.quantization_default import (
    _HIGH_VRAM_DEFAULT,
    _NF4_VRAM_THRESHOLD_GB,
)
from math_lora.types import QUANTIZATION_MODES, HardwareProfile, QuantizationMode

from tests.property.strategies import valid_hardware_profiles


# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------


def _hw_profiles_below_threshold() -> st.SearchStrategy[HardwareProfile]:
    """``HardwareProfile`` strategy filtered to ``vram_per_gpu_gb < 24``.

    The base ``valid_hardware_profiles`` strategy in
    ``tests/property/strategies.py`` already includes the boundary samples
    ``{1, 8, 16, 24, 80}`` plus arbitrary integers in ``[1, 1024]``, so a
    simple ``filter`` gives us a strategy that still hits the consumer-tier
    boundaries (``1``, ``8``, ``16``) at high probability while covering the
    entire sub-threshold range.
    """

    return valid_hardware_profiles().filter(
        lambda hw: hw.vram_per_gpu_gb < _NF4_VRAM_THRESHOLD_GB
    )


def _hw_profiles_at_or_above_threshold() -> st.SearchStrategy[HardwareProfile]:
    """``HardwareProfile`` strategy filtered to ``vram_per_gpu_gb >= 24``.

    Mirror of :func:`_hw_profiles_below_threshold` for the high-VRAM branch.
    Includes the boundary value ``24`` itself (sampled by the base strategy)
    so the strict-``<`` cutoff documented in the resolver is exercised.
    """

    return valid_hardware_profiles().filter(
        lambda hw: hw.vram_per_gpu_gb >= _NF4_VRAM_THRESHOLD_GB
    )


# ---------------------------------------------------------------------------
# Property 5, clause 1: low-VRAM default
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(hardware_profile=_hw_profiles_below_threshold())
def test_property5_low_vram_unset_resolves_to_nf4(
    hardware_profile: HardwareProfile,
) -> None:
    """Validates: Requirements 2.4

    For every ``HardwareProfile`` whose ``vram_per_gpu_gb`` is strictly less
    than ``24`` and an unset (``None``) override, the resolved mode is
    exactly ``"nf4"``. This is the headline default the resolver exists to
    enforce -- consumer GPUs (``8``-``16`` GB) and any other below-threshold
    hardware must land on the QLoRA 4-bit default without operator action.
    """

    # Sanity check on the strategy filter: a property test that only
    # accidentally hits the low-VRAM branch would silently weaken the
    # guarantee, so we re-assert the precondition Hypothesis is supposed
    # to be satisfying for us.
    assert hardware_profile.vram_per_gpu_gb < _NF4_VRAM_THRESHOLD_GB

    resolved = resolve_quantization_mode(hardware_profile, None)

    assert resolved == "nf4", (
        f"low-VRAM default broken: vram_per_gpu_gb="
        f"{hardware_profile.vram_per_gpu_gb} expected 'nf4', got {resolved!r}"
    )


# ---------------------------------------------------------------------------
# Property 5, clause 2: explicit override is respected
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    hardware_profile=valid_hardware_profiles(),
    explicit_mode=st.sampled_from(QUANTIZATION_MODES),
)
def test_property5_explicit_override_is_respected(
    hardware_profile: HardwareProfile,
    explicit_mode: QuantizationMode,
) -> None:
    """Validates: Requirements 2.4

    For every ``HardwareProfile`` (the full ``[1, 1024]`` VRAM range, both
    sides of the 24 GB cutoff) and every member of the ``QuantizationMode``
    literal, supplying that member as ``explicit_mode`` returns the same
    member unchanged. The property must hold *regardless* of VRAM so the
    operator override clause in Requirement 2.4 is enforced everywhere,
    including the case where the override disagrees with what the default
    would otherwise pick (e.g. ``"fp16"`` on an 8 GB GPU, where the
    operator is probing trade-offs the planner will halt on later under
    Requirement 2.6).
    """

    resolved = resolve_quantization_mode(hardware_profile, explicit_mode)

    assert resolved == explicit_mode, (
        f"override clause broken: vram_per_gpu_gb="
        f"{hardware_profile.vram_per_gpu_gb}, explicit_mode={explicit_mode!r}, "
        f"got {resolved!r}"
    )


# ---------------------------------------------------------------------------
# Property 5, clause 3: high-VRAM default mirrors the implementation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(hardware_profile=_hw_profiles_at_or_above_threshold())
def test_property5_high_vram_unset_matches_documented_default(
    hardware_profile: HardwareProfile,
) -> None:
    """Validates: Requirements 2.4

    For every ``HardwareProfile`` whose ``vram_per_gpu_gb`` is at or above
    the documented 24 GB threshold and an unset (``None``) override, the
    resolved mode equals the resolver's documented high-VRAM default
    (currently ``"bf16"``). The property mirrors the implementation rather
    than hard-coding ``"bf16"``: it imports ``_HIGH_VRAM_DEFAULT`` from the
    resolver module so a future change to the high-VRAM default would
    propagate without silently weakening this property.

    Requirement 2.4 itself only fixes the low-VRAM default; the high-VRAM
    default is a documented design choice whose stability we still want to
    pin so that the resolver remains a total function with a single
    documented behaviour on every input.
    """

    assert hardware_profile.vram_per_gpu_gb >= _NF4_VRAM_THRESHOLD_GB

    resolved = resolve_quantization_mode(hardware_profile, None)

    assert resolved == _HIGH_VRAM_DEFAULT, (
        f"high-VRAM default broken: vram_per_gpu_gb="
        f"{hardware_profile.vram_per_gpu_gb} expected "
        f"{_HIGH_VRAM_DEFAULT!r}, got {resolved!r}"
    )
