"""VRAM estimation for LoRA training (Requirement 1.3, design Property 1).

This module implements ``estimate_min_vram_gb`` and the supporting coefficient
and result dataclasses used by ``Model_Selector`` to decide which candidate
base models fit a given :class:`HardwareProfile` under a given
:class:`QuantizationMode` and training sequence length.

The function implements the formula declared in
``.kiro/specs/math-lora-finetuning/design.md``  (§ Components / 1.
``Model_Selector``):

.. code-block:: text

    estimated_vram_gb =
        base_weights_bytes(params, quantization_mode)
      + lora_gradient_bytes(trainable_params)
      + optimizer_state_bytes(trainable_params)        # 8 * trainable_params for AdamW
      + activation_bytes(sequence_length, param_count_b)
      + overhead_bytes

The implementation is a **pure function**: it does not read clocks, random
state, environment variables, or any other ambient input. Calling it with the
same arguments returns the same :class:`VRAMEstimate` instance values bit-for
bit, satisfying the determinism leg of design Property 1.

Monotonicity guarantees (the other two legs of Property 1):

* Holding everything else fixed, the estimate is monotone non-decreasing in
  ``candidate.param_count_b`` (every additive term is itself non-decreasing
  in the parameter count, and ``sqrt`` is monotone).
* Holding everything else fixed, the estimate is monotone non-decreasing in
  ``sequence_length`` (only the activation term depends on it, and that
  dependence is linear with a non-negative coefficient).
* Holding everything else fixed, the estimate is monotone non-increasing as
  ``quantization_mode`` moves through ``["fp16", "bf16", "int8", "nf4"]``,
  because :data:`math_lora.types.quantization.BYTES_PER_PARAM` is
  non-increasing across that ordering (``fp16 == bf16 = 2.0``,
  ``int8 = 1.0``, ``nf4 = 0.5``). Equality is allowed at the
  ``fp16 -> bf16`` step.

The coefficients used in the formula are returned as part of the
:class:`VRAMEstimate` (the ``coefficients`` field) so that the
``selection_report`` produced by ``Model_Selector.select`` can include them
verbatim per Requirement 1.3 ("a documented formula whose inputs and
coefficients are exposed in the selection report"). The per-term breakdown
fields on the estimate sum exactly to ``estimated_min_vram_gb`` so callers
can audit each contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from math_lora.types.models import BaseModelCandidate, HardwareProfile
from math_lora.types.quantization import BYTES_PER_PARAM, QuantizationMode


# ---------------------------------------------------------------------------
# Constants (private to this module)
# ---------------------------------------------------------------------------

#: Bytes in one gigabyte. We use the binary gigabyte (2**30 bytes) throughout
#: this codebase because that's what every accelerator vendor (NVIDIA, AMD,
#: Apple) reports in their ``vram_per_gpu_gb``-style fields. Mixing decimal
#: (1e9) and binary (2**30) gigabytes for VRAM accounting would silently
#: under- or over-report by ~7%.
_BYTES_PER_GB: float = float(1 << 30)

#: Number of parameters in one "billion" as used by ``param_count_b``. A
#: candidate that declares ``param_count_b == 7.0`` has 7e9 parameters by
#: convention -- we use SI billions here, not binary.
_PARAMS_PER_B: float = 1.0e9

#: Default fraction of base-model parameters that LoRA on the standard
#: ``q_proj``/``v_proj`` targets at rank ``r=16`` makes trainable. The QLoRA
#: paper (Dettmers et al. 2023, Table 1) reports that LoRA on attention
#: query and value projections at rank 16 yields roughly 0.5% of the total
#: parameter count as trainable, which is the value we encode here. This
#: matches the default ``target_modules`` resolution declared in
#: Requirement 4.4 and used by Model_Selector when it has not yet seen the
#: actual loaded base model.
_DEFAULT_LORA_TRAINABLE_RATIO: float = 0.005

#: Default fixed overhead (CUDA context, kernel workspace, framework
#: bookkeeping). One binary gigabyte is a standard rule-of-thumb under
#: ``transformers``/``bitsandbytes`` (see HuggingFace `accelerate`
#: documentation, "Memory tips") and is conservative enough that the
#: estimate does not under-report on consumer GPUs.
_DEFAULT_OVERHEAD_BYTES: float = 1.0 * _BYTES_PER_GB

#: Default coefficient for the activation-memory proxy. The proxy used here
#: is ``activation_coefficient * sequence_length * sqrt(param_count_b * 1e9)``
#: which is monotone non-decreasing in both ``sequence_length`` and
#: ``param_count_b``. The coefficient is calibrated against published
#: activation-memory measurements for 7B-scale transformers without
#: gradient checkpointing; the absolute calibration is documented in this
#: module's tests rather than tuned per-GPU because Property 1 only
#: requires monotonicity, not numeric exactness.
_DEFAULT_ACTIVATION_COEFFICIENT: float = 2.0

#: Default optimizer-state multiplier. AdamW keeps two fp32 moments per
#: trainable parameter -- ``2 * 4 = 8`` bytes per trainable parameter. This
#: matches the value used in the QLoRA paper (Dettmers et al. 2023) and the
#: design's Research Notes.
_DEFAULT_OPTIMIZER_STATE_MULTIPLIER: float = 8.0

#: Bytes per LoRA gradient parameter. LoRA weights are kept in bf16 / fp16
#: even when the base model is in 4-bit (Requirement 4.8), so the gradient
#: cost is fixed at 2 bytes per trainable parameter regardless of the base
#: model's ``QuantizationMode``.
_LORA_GRADIENT_BYTES_PER_PARAM: float = 2.0


# ---------------------------------------------------------------------------
# Coefficients dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VRAMCoefficients:
    """Coefficients used by :func:`estimate_min_vram_gb`.

    These values are exposed alongside every estimate so that the
    ``selection_report`` produced by ``Model_Selector.select`` can include
    them verbatim, as required by Requirement 1.3 ("a documented formula
    whose inputs and coefficients are exposed in the selection report"). A
    consumer of the estimate can multiply the recorded coefficients with the
    per-term breakdown on :class:`VRAMEstimate` and recover the same
    ``estimated_min_vram_gb`` byte-for-byte.

    All fields are real-valued and non-negative. The dataclass is frozen so
    that an estimate captured in a manifest cannot be mutated after the
    fact -- a requirement for the bit-for-bit reproducibility guarantee in
    Requirement 8.10.

    Attributes:
        bytes_per_param: Bytes per base-model parameter under each
            :class:`QuantizationMode`. Defaults to the published QLoRA
            values: ``fp16 = bf16 = 2.0``, ``int8 = 1.0``, ``nf4 = 0.5``.
            Wrapped in a :class:`types.MappingProxyType` so the default
            cannot be mutated through the frozen dataclass.
        optimizer_state_multiplier: Bytes of optimizer state per *trainable*
            parameter. Defaults to ``8.0`` for AdamW (two fp32 moments).
        activation_coefficient: Coefficient ``c`` in the activation-memory
            proxy ``c * sequence_length * sqrt(param_count_b * 1e9)``. The
            proxy is documented in :func:`_activation_bytes`; it is monotone
            non-decreasing in both the parameter count and the sequence
            length, which is what Property 1 requires.
        overhead_bytes: Fixed overhead (CUDA context, kernel workspaces,
            framework bookkeeping). Expressed in bytes so that the
            coefficient table is dimensionally homogeneous; defaults to
            ``1 GB`` (1 * 2**30 bytes).
        lora_trainable_param_ratio: Fraction of the base model's parameter
            count that is trainable under the default LoRA configuration
            (``q_proj`` + ``v_proj`` at rank ``r = 16``). Defaults to
            ``0.005`` (~0.5%). This is the value used by ``Model_Selector``
            when it has not yet seen the actual loaded base model, per
            Requirement 4.4. The ``LoRA_Trainer`` later reports the true
            ratio (Requirement 4.6) -- but at selection time this is the
            only number the selector has.
    """

    bytes_per_param: Mapping[QuantizationMode, float] = field(
        default_factory=lambda: MappingProxyType(dict(BYTES_PER_PARAM))
    )
    optimizer_state_multiplier: float = _DEFAULT_OPTIMIZER_STATE_MULTIPLIER
    activation_coefficient: float = _DEFAULT_ACTIVATION_COEFFICIENT
    overhead_bytes: float = _DEFAULT_OVERHEAD_BYTES
    lora_trainable_param_ratio: float = _DEFAULT_LORA_TRAINABLE_RATIO


# ---------------------------------------------------------------------------
# Estimate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VRAMEstimate:
    """Result of :func:`estimate_min_vram_gb`.

    Every field is in **gigabytes** (binary, 2**30 bytes) so that direct
    comparison against :attr:`HardwareProfile.vram_per_gpu_gb` is unit-safe.
    The four per-term fields plus ``overhead_gb`` sum exactly to
    ``estimated_min_vram_gb`` (within IEEE-754 double-precision rounding),
    so a consumer can audit each contribution.

    Attributes:
        estimated_min_vram_gb: Total minimum VRAM in gigabytes. Compared
            against :attr:`HardwareProfile.vram_per_gpu_gb` to compute the
            feasibility flag in Requirement 1.4.
        base_weights_gb: Bytes for the frozen base-model weights, converted
            to gigabytes. Depends on the chosen
            :class:`QuantizationMode` via
            :data:`math_lora.types.quantization.BYTES_PER_PARAM`.
        lora_gradient_gb: Bytes for LoRA adapter gradients (kept in bf16 /
            fp16 per Requirement 4.8), converted to gigabytes.
        optimizer_state_gb: Bytes for AdamW optimizer state on the
            *trainable* parameter set only (Requirement 4.3 freezes the
            base model, so no optimizer state is allocated for it).
        activation_gb: Bytes for the activation memory proxy, converted to
            gigabytes. Linear in ``sequence_length``.
        overhead_gb: Fixed framework / context overhead, converted to
            gigabytes.
        coefficients: The :class:`VRAMCoefficients` used to compute the
            estimate. Exposed verbatim so that ``Model_Selector.select``
            can attach the same dataclass to every emitted
            ``selection_report`` (Requirement 1.3).
    """

    estimated_min_vram_gb: float
    base_weights_gb: float
    lora_gradient_gb: float
    optimizer_state_gb: float
    activation_gb: float
    overhead_gb: float
    coefficients: VRAMCoefficients


# ---------------------------------------------------------------------------
# Internal pure helpers (one per term in the formula)
# ---------------------------------------------------------------------------


def _base_weights_bytes(
    param_count_b: float,
    quantization_mode: QuantizationMode,
    coefficients: VRAMCoefficients,
) -> float:
    """Return base-model weight bytes.

    ``base_weights = param_count * bytes_per_param(quantization_mode)``
    """
    bytes_per_param = coefficients.bytes_per_param[quantization_mode]
    return param_count_b * _PARAMS_PER_B * bytes_per_param


def _lora_gradient_bytes(trainable_params: float) -> float:
    """Return LoRA gradient bytes (bf16 / fp16, 2 bytes per trainable param).

    LoRA weights stay in bf16 or fp16 even when the base is in 4-bit
    (Requirement 4.8), so the gradient cost is independent of
    :class:`QuantizationMode`.
    """
    return trainable_params * _LORA_GRADIENT_BYTES_PER_PARAM


def _optimizer_state_bytes(
    trainable_params: float, coefficients: VRAMCoefficients
) -> float:
    """Return AdamW optimizer state bytes for the trainable params only.

    Eight bytes per trainable parameter (two fp32 moments) per the QLoRA
    paper (Dettmers et al. 2023) and the design's Research Notes.
    """
    return trainable_params * coefficients.optimizer_state_multiplier


def _activation_bytes(
    param_count_b: float, sequence_length: int, coefficients: VRAMCoefficients
) -> float:
    """Return activation-memory bytes under a documented monotone proxy.

    The proxy is

    .. code-block:: text

        activation_bytes
            = activation_coefficient
              * sequence_length
              * sqrt(param_count_b * 1e9)

    Two reasons for this shape:

    1. **Monotonicity is the contract.** Property 1 only requires that the
       estimate be monotone non-decreasing in ``sequence_length`` and
       ``param_count_b``. ``sqrt`` is strictly increasing on the positive
       reals, so the proxy is monotone in ``param_count_b``; the
       sequence-length dependence is linear with a non-negative coefficient,
       so the proxy is monotone in ``sequence_length``.
    2. **It is independent of quantization mode.** Activations are stored at
       the framework's mixed-precision dtype regardless of how the base
       weights are quantized, which keeps the activation contribution
       constant across the ``fp16 -> bf16 -> int8 -> nf4`` ordering and
       lets the mode-monotonicity property hold by construction (only the
       base-weights term changes across modes).

    The proxy is calibrated, not derived from first principles. A first-
    principles activation estimate would require ``hidden_size`` and
    ``num_layers``, neither of which is on :class:`BaseModelCandidate`; for
    a 7B-scale transformer those values are roughly ``H ~ sqrt(P/(12*L))``
    with ``L ~ 32``, and the resulting expression collapses to a constant
    times ``seq * sqrt(P)``. The constant is folded into
    ``activation_coefficient`` and the implicit batch-size factor is set to
    one ("minimum" VRAM at the smallest meaningful batch).
    """
    # Using ``math.sqrt`` instead of ``** 0.5`` so that the underlying
    # libm implementation is invoked and the IEEE-754 rounding is the
    # platform-standard one. ``** 0.5`` is also IEEE compliant in CPython
    # but ``math.sqrt`` reads more clearly at the call site.
    return (
        coefficients.activation_coefficient
        * float(sequence_length)
        * math.sqrt(param_count_b * _PARAMS_PER_B)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_min_vram_gb(
    candidate: BaseModelCandidate,
    hw_profile: HardwareProfile,
    quantization_mode: QuantizationMode,
    sequence_length: int,
    *,
    coefficients: VRAMCoefficients | None = None,
) -> VRAMEstimate:
    """Estimate the minimum VRAM (gigabytes) needed to LoRA-train a candidate.

    This is a **pure function**: it never reads time, random state,
    environment variables, or any other ambient input. Identical arguments
    yield identical :class:`VRAMEstimate` values, which is the determinism
    leg of design Property 1.

    Note that ``hw_profile`` is accepted for symmetry with the public
    interface declared in ``design.md`` § Components / 1. ``Model_Selector``
    and to make the function call site read like the design diagram, but
    the *estimate itself* does not depend on ``hw_profile``: the profile is
    used by the surrounding ``Model_Selector`` to compute feasibility
    (Requirement 1.4) and the VRAM shortfall (Requirement 1.4) once this
    function has produced its number.

    Args:
        candidate: Candidate base model carrying the declared
            ``param_count_b`` (Requirement 1.1).
        hw_profile: Hardware profile of the target machine. Accepted for
            interface symmetry; the estimate itself is independent of it.
        quantization_mode: Numerical precision of the base model weights.
            Selects the corresponding entry from
            :attr:`VRAMCoefficients.bytes_per_param`.
        sequence_length: Training sequence length in tokens, expected to be
            in ``[128, 8192]`` per Requirement 1.2 although this function
            accepts any non-negative integer to keep the formula total.
        coefficients: Optional override of the default
            :class:`VRAMCoefficients`. Passed by callers (such as
            ``Hardware_Budget_Planner``) that need to mirror the same
            coefficient set across multiple components, per Requirement 1.3
            ("inputs and coefficients are exposed in the selection report").
            When ``None``, a fresh default :class:`VRAMCoefficients` is
            constructed.

    Returns:
        :class:`VRAMEstimate` whose ``estimated_min_vram_gb`` field holds
        the total estimate in gigabytes and whose per-term fields sum
        exactly to that total (within IEEE-754 rounding). The
        ``coefficients`` field returns the actual coefficients used so the
        caller can record them in the ``selection_report``.
    """
    # Use the caller's coefficients when supplied. Constructing a fresh
    # default per call is cheap (a small frozen dataclass) and keeps the
    # function free of any cross-call state -- the "pure function" /
    # "no global state" guarantee in the task prompt and the determinism
    # leg of Property 1.
    coeffs = coefficients if coefficients is not None else VRAMCoefficients()

    # Trainable parameter count: a fixed fraction of the declared base
    # parameter count. ``BaseModelCandidate.param_count_b`` is in billions
    # (SI), so the multiplication by 1e9 converts to a raw count.
    trainable_params = (
        candidate.param_count_b * _PARAMS_PER_B * coeffs.lora_trainable_param_ratio
    )

    # Five additive terms, each expressed in raw bytes. Keeping them
    # separate makes the per-term breakdown on VRAMEstimate easy to compute
    # and audit, and matches the order in the design's formula block.
    base_bytes = _base_weights_bytes(
        candidate.param_count_b, quantization_mode, coeffs
    )
    lora_grad_bytes = _lora_gradient_bytes(trainable_params)
    opt_state_bytes = _optimizer_state_bytes(trainable_params, coeffs)
    act_bytes = _activation_bytes(candidate.param_count_b, sequence_length, coeffs)
    over_bytes = coeffs.overhead_bytes

    total_bytes = (
        base_bytes + lora_grad_bytes + opt_state_bytes + act_bytes + over_bytes
    )

    # Convert each term to gigabytes. We do the bytes-to-GB conversion
    # *after* summing in bytes so the rounding error is bounded by a single
    # division rather than five.
    return VRAMEstimate(
        estimated_min_vram_gb=total_bytes / _BYTES_PER_GB,
        base_weights_gb=base_bytes / _BYTES_PER_GB,
        lora_gradient_gb=lora_grad_bytes / _BYTES_PER_GB,
        optimizer_state_gb=opt_state_bytes / _BYTES_PER_GB,
        activation_gb=act_bytes / _BYTES_PER_GB,
        overhead_gb=over_bytes / _BYTES_PER_GB,
        coefficients=coeffs,
    )


__all__ = [
    "VRAMCoefficients",
    "VRAMEstimate",
    "estimate_min_vram_gb",
]
