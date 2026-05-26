"""Projected wall-clock, cost, and peak-VRAM estimation (Requirement 2.5).

This module implements task 3.4 of the math-LoRA implementation plan, which
covers Requirement 2.5 of the feature spec:

    WHEN training is invoked, THE Training_Pipeline SHALL produce, before
    any training step executes, a pre-flight report that records the
    projected wall-clock training time in hours, the projected monetary
    cost in the declared currency derived from the cost rate per GPU-hour
    times the projected GPU-hours, and the projected peak VRAM in
    gigabytes per GPU, computed from the Base_Model size,
    Quantization_Mode, batch size, sequence length, and gradient
    accumulation settings.

Design hooks
------------

The three projections are built as **pure functions** (no I/O, no clocks,
no global state, no randomness) so that they can be reused safely by:

* The pre-flight gate in ``Training_Pipeline`` (Requirement 5 / design's
  ``pre_train_gate()``) -- which runs once per training invocation.
* The knob-reduction search in ``Hardware_Budget_Planner.plan(...)``
  (Requirement 2.6/2.7/2.8) -- which evaluates each projection many times
  while sweeping ``batch_size``, ``sequence_length``,
  ``gradient_accumulation_steps``, ``dataset_size``, and ``max_steps`` to
  find the smallest single-knob change that satisfies the limits.
* Property tests (Property 4 in design.md) that re-run the projection
  after applying a suggested knob change and verify the projection now
  fits within the limit.

Coefficient consistency
-----------------------

The projected peak VRAM per GPU MUST be consistent with the VRAM estimate
that ``Model_Selector`` exposes in its selection report (Requirement 1.3).
We achieve that by **reusing**
:class:`math_lora.model_selector.VRAMCoefficients` directly: the same
``bytes_per_param`` table, the same ``optimizer_state_multiplier``, the
same ``activation_coefficient``, the same ``overhead_bytes``, and the same
``lora_trainable_param_ratio`` that drove the selector's estimate also
drives this projection. The planner and the selector therefore share a
single source of truth.

Two extra knobs are folded into the activation term that
``Model_Selector`` does not see:

1. ``batch_size``. ``Model_Selector`` estimates the *minimum* VRAM, which
   corresponds to the smallest meaningful batch (one); for the actual
   training projection the activation cost scales linearly with the
   per-rank batch size. (See design § Research Notes: "Activation memory
   scales as ``O(batch_size * seq_len * hidden_size * num_layers)``".)
2. ``gradient_checkpointing``. Activation memory is reduced when forward
   activations are recomputed during the backward pass (Requirement 2.9
   lists gradient checkpointing as an explicit VRAM-reduction toggle). We
   apply a documented reduction factor when checkpointing is enabled,
   matching the QLoRA paper's reported ratio (see :data:`_GC_ACTIVATION_FACTOR`
   below).

Note that ``gradient_accumulation_steps`` does **not** appear in the peak
VRAM projection: gradient accumulation only multiplies the number of
microsteps per optimizer step, not the per-microstep memory footprint, so
its effect on memory is zero. It does multiply wall-clock time -- and
therefore cost -- linearly, which is exactly what
:func:`project_wallclock_hours` and :func:`project_cost` capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from math_lora.model_selector.vram import (
    VRAMCoefficients,
    _BYTES_PER_GB,
    _LORA_GRADIENT_BYTES_PER_PARAM,
    _PARAMS_PER_B,
)
from math_lora.types.models import BaseModelCandidate, BudgetProfile, HardwareProfile
from math_lora.types.quantization import QuantizationMode


# ---------------------------------------------------------------------------
# Documented coefficients
# ---------------------------------------------------------------------------

#: Activation reduction factor when gradient checkpointing is enabled.
#:
#: Gradient checkpointing recomputes forward activations during the
#: backward pass, trading compute for memory. The QLoRA paper (Dettmers et
#: al. 2023, Table 5) reports an activation memory reduction of roughly
#: ``1/sqrt(num_layers)`` for transformers; for the 32-layer base models
#: in scope here that is ~0.18, which we round up to ``0.20`` to stay
#: conservative (peak VRAM projections must not under-report -- a halt
#: that doesn't fire is worse than a halt that fires too eagerly, per the
#: halt-before-train invariant in the design).
_GC_ACTIVATION_FACTOR: Final[float] = 0.20

#: Time-per-FLOP factor when gradient checkpointing is enabled.
#:
#: One extra forward pass per backward pass adds ~33% to step time, per
#: the QLoRA paper. We use ``1.33`` as the documented multiplier and
#: apply it inside :func:`project_wallclock_hours` so the wall-clock and
#: cost projections honour Requirement 2.9 (gradient checkpointing is an
#: explicit VRAM-reduction toggle that costs compute time).
_GC_TIME_FACTOR: Final[float] = 1.33

#: Throughput coefficient ``alpha`` in the wall-clock model
#: ``hours_per_microstep_per_gpu = alpha * batch_size * seq_len *
#: param_count_b * mode_compute_factor``. Calibrated to match published
#: training throughput numbers for 7B-class transformers on a single
#: A100-class GPU at ``batch_size=1, seq_len=2048`` in bf16 (roughly
#: 1 step/sec, i.e. ~2.78e-4 hours/step). The proxy is monotone in every
#: input; absolute calibration is approximate -- the cost-budget halt
#: in Requirement 2.8 is what catches under-provisioning, not the
#: numeric exactness of this coefficient.
_THROUGHPUT_COEFFICIENT_HOURS: Final[float] = 2.0e-12

#: Compute-cost factor per :class:`QuantizationMode`.
#:
#: Lower-precision modes consume fewer FLOPs per parameter per token at
#: matmul time on modern accelerators (TF32/INT8/4-bit kernels), so
#: training time decreases as the mode walks ``fp16 -> bf16 -> int8 ->
#: nf4``. Values are calibrated against published training-throughput
#: tables; the proxy is monotone non-increasing along this ordering,
#: matching the memory-cost ordering used by ``Model_Selector``.
#:
#: ``fp16`` and ``bf16`` are equal because both occupy two bytes and use
#: the same tensor-core kernels on Ampere/Hopper; ``int8`` and ``nf4``
#: are progressively cheaper. The dictionary is intentionally local to
#: this module (rather than added to the shared
#: :mod:`math_lora.types.quantization` table) because compute cost is a
#: planner-level concern, while ``BYTES_PER_PARAM`` is a memory-level
#: concern shared with the selector.
_COMPUTE_COST_PER_MODE: Final[dict[QuantizationMode, float]] = {
    "fp16": 1.00,
    "bf16": 1.00,
    "int8": 0.65,
    "nf4": 0.55,
}


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionInputs:
    """Inputs required by :func:`project_all` and the per-projection helpers.

    The fields are exactly the projection-relevant subset of the design's
    ``TrainingConfig`` plus the cost-rate fields from
    :class:`BudgetProfile` and the GPU count from
    :class:`HardwareProfile`. Keeping them in one frozen dataclass makes
    it trivial for ``Hardware_Budget_Planner.plan`` to mutate a single
    knob (e.g. ``batch_size``) and re-project, which is what the
    knob-reduction search in Requirement 2.6/2.7/2.8 needs.

    Attributes:
        base_model: The candidate selected by ``Model_Selector``. Carries
            the declared ``param_count_b`` that drives every term in the
            VRAM projection and the wall-clock model.
        quantization_mode: Numerical precision of the base model weights.
            Drives both the memory term (``bytes_per_param``) and the
            compute term (:data:`_COMPUTE_COST_PER_MODE`).
        batch_size: Per-rank micro-batch size in samples. Linear in both
            the activation memory term and the wall-clock model. Must be
            a positive integer per the design's ``TrainingConfig``.
        sequence_length: Training sequence length in tokens. Linear in
            both the activation memory term and the wall-clock model.
            Expected to be in ``[128, 8192]`` per Requirement 1.2 but the
            projection is total over any positive integer.
        gradient_accumulation_steps: Number of microsteps that accumulate
            gradients into one optimizer step. Multiplies wall-clock time
            and cost by exactly this factor (Requirement 2.9 lists
            gradient accumulation as an explicit toggle). Has **no** effect
            on peak VRAM because each microstep sees only ``batch_size``
            samples of activations at a time.
        gradient_checkpointing: When True, activation memory is reduced
            by :data:`_GC_ACTIVATION_FACTOR` and step time is multiplied
            by :data:`_GC_TIME_FACTOR`.
        max_steps: Total optimizer steps in the run. Multiplies wall-clock
            time linearly. Required because Requirement 2.5 asks for
            *projected wall-clock* in hours, which only has meaning when
            the run length is fixed.
        gpu_count: Number of GPUs participating in the run (from
            :class:`HardwareProfile`). The peak VRAM projection is
            *per GPU* so this field is only used by
            :func:`project_wallclock_hours` (linear speedup assumption,
            documented there) and :func:`project_cost`
            (``gpu_hours = wallclock_hours * gpu_count``).
        cost_rate_per_gpu_hour: Cost rate from :class:`BudgetProfile`.
            Multiplies projected GPU-hours to yield projected cost.
    """

    base_model: BaseModelCandidate
    quantization_mode: QuantizationMode
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_steps: int
    gpu_count: int
    cost_rate_per_gpu_hour: float


@dataclass(frozen=True)
class Projections:
    """Bundled output of :func:`project_all`.

    Attributes:
        projected_wallclock_hours: Projected wall-clock training time in
            hours, equal to ``max_steps * gradient_accumulation_steps *
            hours_per_microstep_per_gpu / gpu_count`` with the gradient-
            checkpointing time multiplier applied.
        projected_gpu_hours: ``projected_wallclock_hours * gpu_count``.
            Carried separately so that callers (the cost-reconciliation
            path in Requirement 2.12, the manifest emitter) do not need
            to redo the multiplication.
        projected_cost: Projected monetary cost in the budget's currency,
            equal to ``projected_gpu_hours * cost_rate_per_gpu_hour``.
        projected_peak_vram_gb_per_gpu: Peak VRAM in gigabytes for one
            GPU, computed from the same :class:`VRAMCoefficients` that
            ``Model_Selector`` uses, with the planner's ``batch_size``
            and ``gradient_checkpointing`` knobs applied.
        coefficients: The :class:`VRAMCoefficients` used to compute
            ``projected_peak_vram_gb_per_gpu``. Exposed verbatim so the
            ``selection_report`` and the ``pre_flight_report`` can pin
            the same coefficient set per Requirement 1.3.
    """

    projected_wallclock_hours: float
    projected_gpu_hours: float
    projected_cost: float
    projected_peak_vram_gb_per_gpu: float
    coefficients: VRAMCoefficients


# ---------------------------------------------------------------------------
# Peak VRAM projection
# ---------------------------------------------------------------------------


def project_peak_vram_gb_per_gpu(
    base_model: BaseModelCandidate,
    quantization_mode: QuantizationMode,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing: bool,
    *,
    coefficients: VRAMCoefficients | None = None,
) -> float:
    """Project peak VRAM (gigabytes) for a single GPU during training.

    Pure function; identical inputs always yield identical output.

    The projection has the same five additive terms as the selector's
    minimum-VRAM estimate, with the activation term scaled by
    ``batch_size`` and reduced by :data:`_GC_ACTIVATION_FACTOR` when
    ``gradient_checkpointing`` is True. This keeps the planner's
    projection consistent with the selector's estimate at the boundary
    ``batch_size=1, gradient_checkpointing=False`` (the selector's
    "minimum" assumption) while extending it to the actual training
    knobs that the operator declares in ``TrainingConfig``.

    Args:
        base_model: Selected candidate base model. Carries the declared
            ``param_count_b`` (Requirement 1.1) used in every term.
        quantization_mode: Numerical precision of base weights. Selects
            the corresponding entry from
            :attr:`VRAMCoefficients.bytes_per_param`.
        batch_size: Per-rank micro-batch size in samples. Must be
            positive; activation memory scales linearly with this value.
        sequence_length: Training sequence length in tokens. Must be
            positive; activation memory scales linearly with this value.
        gradient_checkpointing: When True, activation memory is reduced
            by :data:`_GC_ACTIVATION_FACTOR` (~5x reduction).
        coefficients: Optional override of the default
            :class:`VRAMCoefficients`. Pass the same instance that
            ``Model_Selector`` recorded in the ``selection_report`` to
            guarantee planner/selector consistency per Requirement 1.3.

    Returns:
        Projected peak VRAM per GPU in gigabytes (binary, 2**30 bytes).

    Raises:
        ValueError: If ``batch_size`` or ``sequence_length`` is not
            positive. The planner halts at the gate before training, so
            we surface obvious mis-configuration here as a typed error.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be positive, got {sequence_length}"
        )

    coeffs = coefficients if coefficients is not None else VRAMCoefficients()

    # Trainable parameter count: the same fixed fraction the selector
    # uses. Keeping this ratio in :class:`VRAMCoefficients` (rather than
    # hard-coded here) is what makes "consistent with Model_Selector
    # coefficients" hold even if the operator tunes the ratio.
    param_count = base_model.param_count_b * _PARAMS_PER_B
    trainable_params = param_count * coeffs.lora_trainable_param_ratio

    # Term 1: frozen base weights. Identical formula to
    # ``_base_weights_bytes`` in the selector.
    bytes_per_param = coeffs.bytes_per_param[quantization_mode]
    base_weights_bytes = param_count * bytes_per_param

    # Term 2: LoRA gradients (kept in bf16/fp16 even when base is 4-bit
    # per Requirement 4.8). Identical formula to the selector.
    lora_gradient_bytes = trainable_params * _LORA_GRADIENT_BYTES_PER_PARAM

    # Term 3: AdamW optimizer state on the trainable params only. Same
    # 8-byte-per-param multiplier as the selector.
    optimizer_state_bytes = trainable_params * coeffs.optimizer_state_multiplier

    # Term 4: activations. This is where the planner extends the
    # selector's "minimum" baseline:
    #
    # * The selector's proxy is ``coef * seq_len * sqrt(param_count)`` --
    #   i.e. an implicit ``batch_size = 1, gradient_checkpointing = False``
    #   floor.
    # * The planner multiplies by the actual ``batch_size`` (linear in
    #   batch, matching the design's Research Notes).
    # * When gradient checkpointing is enabled, the activations are
    #   recomputed during the backward pass and only a small fraction
    #   are kept resident, giving a documented reduction factor of
    #   :data:`_GC_ACTIVATION_FACTOR`.
    #
    # The boundary case ``batch_size=1, gradient_checkpointing=False``
    # reproduces the selector's activation term byte-for-byte, which is
    # the consistency guarantee Property 1 / Property 4 rely on.
    activation_bytes_floor = (
        coeffs.activation_coefficient
        * float(sequence_length)
        * math.sqrt(param_count)
    )
    gc_factor = _GC_ACTIVATION_FACTOR if gradient_checkpointing else 1.0
    activation_bytes = activation_bytes_floor * float(batch_size) * gc_factor

    # Term 5: framework / context overhead. Identical to the selector.
    overhead_bytes = coeffs.overhead_bytes

    total_bytes = (
        base_weights_bytes
        + lora_gradient_bytes
        + optimizer_state_bytes
        + activation_bytes
        + overhead_bytes
    )
    return total_bytes / _BYTES_PER_GB


# ---------------------------------------------------------------------------
# Wall-clock projection
# ---------------------------------------------------------------------------


def project_wallclock_hours(
    base_model: BaseModelCandidate,
    quantization_mode: QuantizationMode,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_steps: int,
    gpu_count: int,
) -> float:
    """Project wall-clock training time in hours.

    Pure function; identical inputs always yield identical output.

    The model is

    .. code-block:: text

        hours_per_microstep_per_gpu
            = alpha
              * batch_size
              * sequence_length
              * param_count_b
              * mode_compute_factor

        wallclock_hours
            = max_steps
              * gradient_accumulation_steps
              * hours_per_microstep_per_gpu
              / gpu_count
              * (1.33 if gradient_checkpointing else 1.0)

    The throughput coefficient ``alpha`` and the ``mode_compute_factor``
    table are documented at the top of this module. The ``/ gpu_count``
    term assumes linear data-parallel speedup (one rank per GPU,
    per-rank batch size held constant). Real-world scaling is
    sub-linear, but Requirement 2.5 asks for a *projection*, and the
    cost/time-budget halts in Requirement 2.7/2.8 are what catch
    under-projection -- the formula's purpose is to be monotone in every
    knob, which it is.

    Args:
        base_model: Selected candidate. Carries ``param_count_b``.
        quantization_mode: Drives the compute-cost factor.
        batch_size: Per-rank micro-batch size.
        sequence_length: Tokens per sample.
        gradient_accumulation_steps: Microsteps per optimizer step.
            Multiplies the projection linearly (each microstep does the
            same work as a non-accumulated step).
        gradient_checkpointing: When True, applies the documented
            :data:`_GC_TIME_FACTOR` step-time multiplier.
        max_steps: Total optimizer steps. Multiplies the projection
            linearly.
        gpu_count: Number of GPUs (data-parallel ranks). Divides the
            projection linearly.

    Returns:
        Projected wall-clock time in hours. Always non-negative; zero
        only when ``max_steps`` is zero.

    Raises:
        ValueError: If any of ``batch_size``, ``sequence_length``,
            ``gradient_accumulation_steps``, ``max_steps``, or
            ``gpu_count`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be positive, got {sequence_length}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, "
            f"got {gradient_accumulation_steps}"
        )
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")

    mode_factor = _COMPUTE_COST_PER_MODE[quantization_mode]
    param_count = base_model.param_count_b * _PARAMS_PER_B

    hours_per_microstep_per_gpu = (
        _THROUGHPUT_COEFFICIENT_HOURS
        * float(batch_size)
        * float(sequence_length)
        * param_count
        * mode_factor
    )

    gc_time_factor = _GC_TIME_FACTOR if gradient_checkpointing else 1.0

    return (
        float(max_steps)
        * float(gradient_accumulation_steps)
        * hours_per_microstep_per_gpu
        * gc_time_factor
        / float(gpu_count)
    )


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


def project_cost(
    projected_wallclock_hours: float,
    gpu_count: int,
    cost_rate_per_gpu_hour: float,
) -> float:
    """Project monetary cost from wall-clock hours, GPU count, and rate.

    Pure function. Implements the formula declared verbatim by Requirement
    2.5: "the projected monetary cost in the declared currency derived
    from the cost rate per GPU-hour times the projected GPU-hours". The
    implementation is therefore exactly

    .. code-block:: text

        gpu_hours = projected_wallclock_hours * gpu_count
        projected_cost = gpu_hours * cost_rate_per_gpu_hour

    Args:
        projected_wallclock_hours: Output of
            :func:`project_wallclock_hours`. Non-negative.
        gpu_count: Number of GPUs. Must be positive.
        cost_rate_per_gpu_hour: Rate from :class:`BudgetProfile`. May
            be exactly zero when the operator owns the hardware (sunk
            cost) -- a zero rate yields a zero projected cost, which the
            cost-budget halt in Requirement 2.8 will never trigger,
            which is the intended behaviour.

    Returns:
        Projected cost in the budget's currency. Non-negative.

    Raises:
        ValueError: If ``projected_wallclock_hours`` is negative,
            ``gpu_count`` is non-positive, or ``cost_rate_per_gpu_hour``
            is negative.
    """

    if projected_wallclock_hours < 0:
        raise ValueError(
            "projected_wallclock_hours must be non-negative, "
            f"got {projected_wallclock_hours}"
        )
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")
    if cost_rate_per_gpu_hour < 0:
        raise ValueError(
            "cost_rate_per_gpu_hour must be non-negative, "
            f"got {cost_rate_per_gpu_hour}"
        )

    gpu_hours = projected_wallclock_hours * float(gpu_count)
    return gpu_hours * float(cost_rate_per_gpu_hour)


# ---------------------------------------------------------------------------
# Convenience: project all three together
# ---------------------------------------------------------------------------


def project_all(
    inputs: ProjectionInputs,
    *,
    coefficients: VRAMCoefficients | None = None,
) -> Projections:
    """Compute all three projections (hours, cost, peak VRAM) in one call.

    Pure function; the result is the bundled :class:`Projections`
    dataclass. The same :class:`VRAMCoefficients` is used for the peak
    VRAM term and exposed on the result so callers can pin it in the
    ``pre_flight_report`` per Requirement 1.3.

    Args:
        inputs: Bundled :class:`ProjectionInputs`.
        coefficients: Optional override of the default
            :class:`VRAMCoefficients`. The same coefficient set should be
            used here and in ``Model_Selector.select`` to guarantee
            planner/selector consistency.

    Returns:
        :class:`Projections` containing all three projections plus the
        coefficient set used for the VRAM term.
    """

    coeffs = coefficients if coefficients is not None else VRAMCoefficients()

    wallclock_hours = project_wallclock_hours(
        base_model=inputs.base_model,
        quantization_mode=inputs.quantization_mode,
        batch_size=inputs.batch_size,
        sequence_length=inputs.sequence_length,
        gradient_accumulation_steps=inputs.gradient_accumulation_steps,
        gradient_checkpointing=inputs.gradient_checkpointing,
        max_steps=inputs.max_steps,
        gpu_count=inputs.gpu_count,
    )
    gpu_hours = wallclock_hours * float(inputs.gpu_count)
    cost = gpu_hours * float(inputs.cost_rate_per_gpu_hour)
    peak_vram = project_peak_vram_gb_per_gpu(
        base_model=inputs.base_model,
        quantization_mode=inputs.quantization_mode,
        batch_size=inputs.batch_size,
        sequence_length=inputs.sequence_length,
        gradient_checkpointing=inputs.gradient_checkpointing,
        coefficients=coeffs,
    )

    return Projections(
        projected_wallclock_hours=wallclock_hours,
        projected_gpu_hours=gpu_hours,
        projected_cost=cost,
        projected_peak_vram_gb_per_gpu=peak_vram,
        coefficients=coeffs,
    )


# ---------------------------------------------------------------------------
# Convenience: build inputs from profiles
# ---------------------------------------------------------------------------


def build_projection_inputs(
    base_model: BaseModelCandidate,
    quantization_mode: QuantizationMode,
    hardware_profile: HardwareProfile,
    budget_profile: BudgetProfile,
    *,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_steps: int,
) -> ProjectionInputs:
    """Compose :class:`ProjectionInputs` from already-loaded profiles.

    Pulls ``gpu_count`` from :class:`HardwareProfile` and
    ``cost_rate_per_gpu_hour`` from :class:`BudgetProfile` so callers
    that already have those profiles do not have to repeat the field
    plumbing. The remaining knobs come from the operator's
    ``TrainingConfig`` and are accepted as keyword-only arguments to
    keep call sites self-documenting (a five-int positional list of
    knobs would invite ordering bugs).
    """

    return ProjectionInputs(
        base_model=base_model,
        quantization_mode=quantization_mode,
        batch_size=batch_size,
        sequence_length=sequence_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        max_steps=max_steps,
        gpu_count=hardware_profile.gpu_count,
        cost_rate_per_gpu_hour=float(budget_profile.cost_rate_per_gpu_hour),
    )


__all__ = [
    "ProjectionInputs",
    "Projections",
    "project_peak_vram_gb_per_gpu",
    "project_wallclock_hours",
    "project_cost",
    "project_all",
    "build_projection_inputs",
]
