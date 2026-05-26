"""VRAM-sampling cadence helper for the ``Training_Pipeline`` (Requirement 2.10).

This module implements task 3.7 of the math-LoRA implementation plan, which
covers Requirement 2.10 of the feature spec:

    WHILE training is in progress, THE Training_Pipeline SHALL sample peak
    VRAM usage per GPU at least once per 100 training steps within a window
    of 100 consecutive steps, and SHALL record each sample together with
    its training step index to the Run_Manifest.

The corresponding correctness statement, design Property 6, formalises the
cadence as a coverage invariant:

    For any simulated training run of N steps with N >= 100, for any
    contiguous window of 100 step indices [k, k+100) within [0, N), the
    recorded vram_samples SHALL contain at least one sample whose step
    index lies in that window.

Public surface
--------------

* :class:`VRAMSample` -- a frozen pydantic model holding ``(step,
  peak_vram_gb_per_gpu)`` exactly as the design's Run_Manifest field
  requires (``vram_samples: list[{ step: int, peak_vram_gb_per_gpu:
  list[float] }]``).
* :class:`VRAMSamplingPolicy` -- a frozen pydantic model declaring how
  often to sample (``interval_steps`` in ``[1, 100]``) and whether to
  sample at step zero (``sample_at_step_zero``).
* :func:`should_sample` -- a pure function answering "does this step
  trigger a sample under this policy?" used by both the property test
  and the stateful scheduler. Pure so callers can reason about cadence
  without instantiating any state.
* :class:`VRAMSampleScheduler` -- the stateful helper that the
  ``Training_Pipeline`` uses at runtime: at every step it consults
  :func:`should_sample`, calls the injected peak-VRAM measurement
  callable, appends the resulting :class:`VRAMSample` to its internal
  buffer, and exposes the buffer for the manifest writer.
* :data:`DEFAULT_VRAM_SAMPLING_POLICY` -- the canonical default policy
  (``interval_steps=100``, ``sample_at_step_zero=True``) which is
  exactly the boundary that makes Property 6 hold: any 100-step half-open
  window contains at least one multiple of 100, so sampling at multiples
  of 100 (plus step 0) covers every window.
* :func:`null_peak_vram_measurement` -- the documented stub measurement
  the task prompt asks for ("Default callable can be a stub returning
  0.0 -- actual measurement is wired by Training_Pipeline (task 7.17)
  later"). Real CUDA / ROCm / Metal measurement lives in task 7.17.

Mocking contract
----------------

The peak-VRAM measurement is **always** an injected callable. There is no
hidden global accessor and no environment-detection fallback in this
module. That keeps the helper:

* deterministic under test (a fake measurement returns whatever the test
  wants) -- which is the design's stated mocking strategy ("The GPU memory
  measurement (for VRAM sampling cadence testing)" is listed as a mocked
  dependency in design.md § Testing Strategy / Mocks),
* free of CUDA / accelerator imports at module load time -- which keeps
  the planner package importable on a CPU-only developer machine,
* aligned with how task 7.17 will wire the real measurement -- the
  Training_Pipeline will pass a ``functools.partial`` over
  ``torch.cuda.max_memory_allocated`` (or the ROCm / Metal equivalent)
  reset between samples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)


# ---------------------------------------------------------------------------
# Documented bounds and defaults
# ---------------------------------------------------------------------------

#: Maximum allowed value of ``interval_steps``. Fixed by Requirement 2.10:
#: "at least once per 100 training steps within a window of 100 consecutive
#: steps". Sampling more frequently is always allowed; sampling less
#: frequently than every 100 steps would let a 100-wide window slip past
#: with no sample, violating Property 6.
MAX_SAMPLING_INTERVAL_STEPS: Final[int] = 100

#: Type alias for the peak-VRAM measurement callable. The callable receives
#: the current training step index (so a sophisticated implementation can,
#: for example, log per-step profiling alongside the measurement) and
#: returns one non-negative float per GPU on the host. The number of
#: entries must equal ``HardwareProfile.gpu_count`` -- the scheduler
#: validates this on every call so a misconfigured measurement surfaces
#: at the first sample rather than corrupting the manifest silently.
PeakVRAMMeasurement = Callable[[int], Sequence[float]]


def null_peak_vram_measurement(_step: int) -> tuple[float, ...]:
    """Return ``(0.0,)`` regardless of input.

    This is the "stub returning 0.0" called out in the task prompt. It is
    intentionally a one-element tuple (not zero-length) because:

    * Requirement 1.2 declares ``gpu_count >= 1`` -- a math-LoRA run
      always uses at least one GPU.
    * The Run_Manifest schema declares ``peak_vram_gb_per_gpu`` as a
      ``list[float]`` -- an empty list would be schema-legal but
      semantically wrong, since "no GPUs" cannot occur during a training
      run.

    Real CUDA / ROCm / Metal measurement is wired in task 7.17 of the
    implementation plan; this stub keeps the helper self-contained for
    unit tests and for early prototyping.
    """

    return (0.0,)


# ---------------------------------------------------------------------------
# Pydantic models (VRAMSample, VRAMSamplingPolicy)
# ---------------------------------------------------------------------------


class VRAMSample(BaseModel):
    """One ``(step, peak_vram_gb_per_gpu)`` record for the Run_Manifest.

    Mirrors the manifest field declared in the design's Data Models
    section::

        vram_samples: list[{ step: int, peak_vram_gb_per_gpu: list[float] }]

    The model is frozen and forbids extra fields so a sample appended
    to the in-memory buffer cannot be mutated or extended after the
    fact. That is a requirement for the bit-for-bit reproducibility
    guarantee in Requirement 8.10 -- if a downstream consumer could
    rewrite a sample, two manifests with identical inputs could diverge.

    Attributes:
        step: The 0-indexed training step at which the sample was taken.
            Property 6 references step indices in ``[0, N)``, so step 0
            is a valid value (it corresponds to the first sample of the
            run when ``sample_at_step_zero`` is ``True``).
        peak_vram_gb_per_gpu: One non-negative float per GPU, in
            gigabytes. Stored as a ``tuple`` rather than a ``list`` so
            that the model remains hashable and immutable; pydantic
            still serialises it as a JSON array, matching the manifest
            schema. Length is constrained to ``>= 1`` because
            Requirement 1.2 declares ``gpu_count >= 1``.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
    )

    step: Annotated[StrictInt, Field(ge=0)]
    peak_vram_gb_per_gpu: Annotated[
        tuple[Annotated[StrictFloat | StrictInt, Field(ge=0.0)], ...],
        Field(min_length=1),
    ]


class VRAMSamplingPolicy(BaseModel):
    """Cadence policy consumed by :class:`VRAMSampleScheduler`.

    The policy is a tiny pydantic model rather than a bare integer so
    that:

    * field-level validation surfaces with the offending field named --
      consistent with the design's *Error Handling* table for every
      other config-shaped object in this pipeline,
    * future cadence knobs (e.g. an ``adaptive`` flag, or a per-rank
      stagger offset) can be added without breaking the public signature
      of :class:`VRAMSampleScheduler`,
    * the resolved policy can be embedded verbatim in the
      :class:`~Run_Manifest` alongside the samples themselves, so the
      manifest reader can confirm "yes, the cadence really was what we
      think it was".

    Attributes:
        interval_steps: Sampling period in training steps. Constrained to
            ``[1, 100]`` because Requirement 2.10 caps the gap between
            samples at 100 steps; values above 100 would let a window
            slip past with no sample, violating Property 6. The default
            value ``100`` is the largest legal value -- "as cheap as
            possible while still correct" -- so default-configured
            training runs spend the minimum overhead on VRAM probing.
        sample_at_step_zero: Whether the scheduler produces a sample at
            step 0. Required to be ``True`` to make Property 6 hold for
            a window that starts at ``k = 0``: with
            ``interval_steps == 100`` the next sample after 0 lands at
            step 100, so without a step-0 sample the window
            ``[0, 100)`` would be empty. Operators may set it to
            ``False`` only if they pair it with ``interval_steps < 100``
            so the property still holds; this is enforced at policy
            construction time by :meth:`_validate_property_6_holds`.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
    )

    interval_steps: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_SAMPLING_INTERVAL_STEPS),
    ]
    sample_at_step_zero: StrictBool = True

    @model_validator(mode="after")
    def _validate_property_6_holds(self) -> "VRAMSamplingPolicy":
        """Reject policies that would violate design Property 6.

        Property 6 states that every contiguous 100-step window must
        contain at least one sample. With ``sample_at_step_zero=False``
        and ``interval_steps=100``, the first sample lands at step 100
        and the window ``[0, 100)`` -- a legal window per Property 6
        -- is empty. The policy is rejected in that combination.
        """

        if (
            not self.sample_at_step_zero
            and self.interval_steps >= MAX_SAMPLING_INTERVAL_STEPS
        ):
            raise ValueError(
                "VRAMSamplingPolicy with sample_at_step_zero=False requires "
                f"interval_steps < {MAX_SAMPLING_INTERVAL_STEPS} so the window "
                "[0, 100) still contains a sample (Requirement 2.10 / "
                "design Property 6)."
            )
        return self


#: Default policy used when callers do not supply one.
#:
#: ``interval_steps=100`` is the maximum legal value (Requirement 2.10).
#: ``sample_at_step_zero=True`` ensures the window ``[0, 100)`` is
#: covered. This default is the cheapest correct cadence: one sample per
#: 100 steps plus one at the start of the run.
DEFAULT_VRAM_SAMPLING_POLICY: Final[VRAMSamplingPolicy] = VRAMSamplingPolicy(
    interval_steps=MAX_SAMPLING_INTERVAL_STEPS,
    sample_at_step_zero=True,
)


# ---------------------------------------------------------------------------
# Pure cadence function
# ---------------------------------------------------------------------------


def should_sample(step: int, policy: VRAMSamplingPolicy) -> bool:
    """Return ``True`` iff ``step`` triggers a sample under ``policy``.

    This is a **pure function**: no I/O, no global state, no time-of-day
    dependency. The same ``(step, policy)`` always returns the same
    answer, which is what design Property 6 (a universal statement
    quantified over all step indices) needs to be testable.

    The rule is::

        should_sample(step, policy) ==
            (step == 0 and policy.sample_at_step_zero)
            or (step > 0 and step % policy.interval_steps == 0)

    A worked example with ``interval_steps=100, sample_at_step_zero=True``:

    +------+------------------+
    | step | should_sample    |
    +======+==================+
    |    0 | True             |
    +------+------------------+
    |    1 | False            |
    +------+------------------+
    |   99 | False            |
    +------+------------------+
    |  100 | True             |
    +------+------------------+
    |  150 | False            |
    +------+------------------+
    |  200 | True             |
    +------+------------------+

    Why this guarantees Property 6 with ``interval_steps == 100``: any
    contiguous window of 100 step indices ``[k, k+100)`` with ``k >= 0``
    contains exactly one multiple of 100 (because 100 consecutive
    integers contain exactly one multiple of any divisor of 100). When
    ``k == 0`` that multiple is 0 itself; the policy's
    ``sample_at_step_zero=True`` clause ensures the sample is taken there.
    For ``k >= 1`` the multiple is ``ceil(k/100) * 100`` which is in
    ``[k, k+100)`` and is sampled by the ``step % 100 == 0`` clause.

    Args:
        step: A 0-indexed training step. Negative values are rejected as
            ``ValueError`` because the rest of the pipeline never feeds
            negative step indices to this helper -- the spec uses ``[0,
            N)`` exclusively.
        policy: The cadence policy. Already validated at construction
            time, so the function does not re-check field ranges here.

    Returns:
        ``True`` if a sample should be taken at this step,
        ``False`` otherwise.

    Raises:
        ValueError: If ``step`` is negative.
    """

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if step == 0:
        return policy.sample_at_step_zero

    # ``step > 0`` from here. Modular arithmetic with the validated
    # ``interval_steps`` (``>= 1``) is total -- no divide-by-zero risk.
    return step % policy.interval_steps == 0


# ---------------------------------------------------------------------------
# Stateful scheduler
# ---------------------------------------------------------------------------


class VRAMSampleScheduler:
    """Stateful helper that records :class:`VRAMSample` events during training.

    The scheduler is the runtime counterpart to :func:`should_sample`. The
    ``Training_Pipeline`` (task 7.17 of the implementation plan) calls
    :meth:`maybe_sample` at every training step; the scheduler decides
    whether to invoke the injected measurement callable and, if so,
    builds a :class:`VRAMSample` and appends it to its internal buffer.
    The buffer is later read by :class:`Experiment_Tracker` to populate
    the manifest's ``vram_samples`` field.

    Why stateful rather than a generator
    ------------------------------------

    A generator-style helper would couple sampling cadence to iteration
    of the training loop, which complicates resume-from-checkpoint
    (Requirement 5.3) -- the resumed run would need a fresh generator
    primed to the resumed step. A stateful object that the training
    loop pokes once per step is symmetrical with how the loop already
    interacts with the optimizer, the scheduler, and the experiment
    tracker, and it makes :meth:`maybe_sample` trivially safe to call
    after a checkpoint resume: the previous samples are loaded from the
    checkpoint and appended to the buffer, the scheduler resumes from
    the resumed step counter, and Property 6 still holds because the
    cadence rule depends only on the absolute step index.

    Attributes:
        policy: The cadence policy in force.
        measurement_fn: The injected peak-VRAM measurement callable.
        gpu_count: Expected number of entries in each measurement
            result. ``None`` means "accept any positive length"; pass
            an integer to enforce that ``measurement_fn`` returns
            exactly that many floats. Recommended: pass
            ``HardwareProfile.gpu_count`` so a misconfigured measurement
            fails fast.
    """

    def __init__(
        self,
        policy: VRAMSamplingPolicy = DEFAULT_VRAM_SAMPLING_POLICY,
        *,
        measurement_fn: PeakVRAMMeasurement = null_peak_vram_measurement,
        gpu_count: int | None = None,
        initial_samples: Sequence[VRAMSample] = (),
    ) -> None:
        # Validate ``gpu_count`` ourselves (rather than relying on a
        # later AssertionError) so that misconfigurations surface at
        # construction time. Negative or zero values are nonsensical
        # because Requirement 1.2 declares ``gpu_count >= 1``.
        if gpu_count is not None and gpu_count < 1:
            raise ValueError(
                f"gpu_count must be >= 1 if provided, got {gpu_count}"
            )

        self._policy = policy
        self._measurement_fn = measurement_fn
        self._gpu_count = gpu_count

        # Internal buffer. We accept ``initial_samples`` so that a
        # resume-from-checkpoint flow can rehydrate the scheduler with
        # the samples already taken before the checkpoint -- without
        # this, the resumed run's manifest would lose the pre-resume
        # samples (Requirement 5.3 + Requirement 8.4).
        self._samples: list[VRAMSample] = list(initial_samples)

    # ---- read-only views ----------------------------------------------------

    @property
    def policy(self) -> VRAMSamplingPolicy:
        """The cadence policy in force."""
        return self._policy

    @property
    def samples(self) -> tuple[VRAMSample, ...]:
        """Snapshot of the recorded samples in insertion order.

        Returned as a tuple so callers cannot mutate the scheduler's
        internal buffer -- the scheduler is the only writer of
        ``vram_samples`` for the manifest.
        """
        return tuple(self._samples)

    def __len__(self) -> int:
        """Number of samples recorded so far."""
        return len(self._samples)

    # ---- step-driven API ----------------------------------------------------

    def maybe_sample(self, step: int) -> VRAMSample | None:
        """Sample at ``step`` if the policy says so.

        This is the method the ``Training_Pipeline`` calls once per
        training step (task 7.17). It is total -- it never raises on a
        valid step counter -- so the training loop does not need to
        guard the call.

        Args:
            step: A 0-indexed training step. Must be non-negative;
                negative values raise ``ValueError`` (delegated to
                :func:`should_sample`).

        Returns:
            The newly-recorded :class:`VRAMSample` if a sample was
            taken, ``None`` otherwise. Returning the sample (rather
            than only ``None``-or-not) lets the caller mirror it into
            a logging stream without re-reading the buffer.
        """

        if not should_sample(step, self._policy):
            return None

        return self._record_sample(step)

    def force_sample(self, step: int) -> VRAMSample:
        """Take a sample at ``step`` regardless of the cadence policy.

        Used by the ``Training_Pipeline`` to record a final sample at the
        last step of a run (so the manifest's ``vram_samples`` always
        ends with the terminal step's peak-VRAM, even if that step is
        not a multiple of ``interval_steps``). Property 6 is unaffected
        because adding samples can only *increase* coverage.
        """

        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        return self._record_sample(step)

    # ---- internals ---------------------------------------------------------

    def _record_sample(self, step: int) -> VRAMSample:
        """Invoke ``measurement_fn`` and append a :class:`VRAMSample`."""

        raw = self._measurement_fn(step)

        # Coerce the measurement into a tuple of floats up front. The
        # callable's contract is ``Sequence[float]`` but we accept any
        # iterable shape here because real measurement functions
        # (``torch.cuda.max_memory_allocated`` over a list of devices)
        # return generators in some torch versions.
        peak_per_gpu = tuple(float(x) for x in raw)

        if self._gpu_count is not None and len(peak_per_gpu) != self._gpu_count:
            raise ValueError(
                "peak-VRAM measurement returned "
                f"{len(peak_per_gpu)} value(s), expected {self._gpu_count} "
                "(one per GPU per Requirement 2.10)"
            )
        if len(peak_per_gpu) < 1:
            # Belt-and-braces: even when ``gpu_count`` is ``None`` we
            # require at least one entry, because the manifest schema
            # requires ``peak_vram_gb_per_gpu`` to be non-empty.
            raise ValueError(
                "peak-VRAM measurement returned an empty sequence; "
                "expected at least one entry (Requirement 1.2 declares "
                "gpu_count >= 1)"
            )

        sample = VRAMSample(step=step, peak_vram_gb_per_gpu=peak_per_gpu)
        self._samples.append(sample)
        return sample


__all__ = [
    "DEFAULT_VRAM_SAMPLING_POLICY",
    "MAX_SAMPLING_INTERVAL_STEPS",
    "PeakVRAMMeasurement",
    "VRAMSample",
    "VRAMSampleScheduler",
    "VRAMSamplingPolicy",
    "null_peak_vram_measurement",
    "should_sample",
]
