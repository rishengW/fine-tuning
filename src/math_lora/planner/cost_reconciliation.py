"""End-of-run cost reconciliation for the ``Hardware_Budget_Planner`` (Req 2.12).

This module implements task 3.9 of the implementation plan, which covers
Requirement 2.12 of the feature spec:

    WHERE rented cloud GPUs are used, THE Training_Pipeline SHALL emit, at
    the end of the run, a cost reconciliation that records the projected
    cost from criterion 5, the actual elapsed GPU-hours, the actual cost
    computed as actual elapsed GPU-hours times the declared cost rate, and
    the absolute and percentage difference between projected and actual
    cost.

The arithmetic is fixed by **design.md Property 7**:

* ``actual_cost == actual_gpu_hours * cost_rate_per_gpu_hour``
* ``absolute_diff == abs(actual_cost - projected_cost)``  -- always non-negative
* ``pct_diff == (absolute_diff / projected_cost) * 100``  when ``projected_cost > 0``
* ``pct_diff is None`` when ``projected_cost == 0``  -- documented sentinel

The ``pct_diff`` sentinel for the zero-projected-cost edge case is needed
because percentage change against a zero baseline is mathematically
undefined. We chose ``None`` (rather than ``inf``, ``nan``, or ``0.0``) so
that downstream consumers (``Experiment_Tracker`` manifest, log lines, JSON
serialisation) cannot silently treat the sentinel as a real number; serializing
``None`` to JSON produces ``null`` which is unambiguously distinct from a real
percentage value.

Pure-function design
--------------------

Per task 3.9 the function MUST be pure: no I/O, no global state, deterministic.
The cloud-vs-local emission gating ("Hardware_Profile.deployment == 'cloud'")
is handled later by ``Training_Pipeline`` (task 12.3); this module only
delivers the arithmetic and a structured result type.

Decoupling from ``PreFlightReport``
-----------------------------------

``PreFlightReport`` is constructed in task 3.5 (not yet implemented at the
time of this task). To avoid a forward dependency on a class that does not
yet exist, this module accepts any object exposing two attributes via the
:class:`PreFlightCostInputs` :class:`typing.Protocol`:

* ``projected_cost: float``  -- from Requirement 2.5 / design data model
* ``cost_rate_per_gpu_hour: float``  -- from ``BudgetProfile`` (Req 2.2)

When task 3.5 lands, ``PreFlightReport`` need only expose those two
attributes (the design data model already includes ``projected_cost`` and
the ``cost_rate_per_gpu_hour`` is naturally carried over from the
``BudgetProfile`` consumed by ``plan(...)``); it will then satisfy this
``Protocol`` automatically without any code change here.

The optional ``projected_currency`` attribute, if present, is propagated to
the result so the manifest entry stays self-describing; absent it, the
result's ``currency`` field is ``None``.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from math_lora.types.errors import SchemaValidationError


# ---------------------------------------------------------------------------
# Input protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PreFlightCostInputs(Protocol):
    """Minimum read-only interface ``reconcile_cost`` requires from ``pre_flight``.

    The protocol is :func:`runtime_checkable` so callers (and tests) can
    do ``isinstance(obj, PreFlightCostInputs)`` if they need a defensive
    guard. In practice we just rely on attribute access -- duck typing is
    sufficient because the function is small, pure, and explicitly typed.

    Attributes:
        projected_cost: Pre-flight projected monetary cost (Requirement
            2.5). Must be ``>= 0`` -- a zero projection is a documented
            edge case (e.g. locally-owned hardware modelled with
            ``cost_rate_per_gpu_hour == 0``) handled by the ``pct_diff``
            sentinel below.
        cost_rate_per_gpu_hour: Declared cost rate from the
            ``BudgetProfile`` (Requirement 2.2). Must be ``>= 0`` per the
            ``BudgetProfile`` schema; values are not coerced or rescaled
            here.

    Notes:
        ``projected_currency`` is intentionally NOT part of this protocol.
        Currency is ancillary metadata for reporting; the function keeps
        it optional via ``getattr`` rather than promoting it to a
        required attribute, which would make the protocol harder to
        satisfy from minimal test stubs.
    """

    @property
    def projected_cost(self) -> float: ...

    @property
    def cost_rate_per_gpu_hour(self) -> float: ...


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class CostReconciliation(BaseModel):
    """Structured end-of-run cost reconciliation record (Requirement 2.12).

    Frozen so the value can be embedded in the ``Run_Manifest``
    (design § Run_Manifest, ``cost_reconciliation`` field) without any
    risk of post-hoc mutation by downstream consumers.

    All four arithmetic fields are computed by :func:`reconcile_cost` and
    are deliberately stored as ``float`` rather than ``Decimal``: the
    reconciliation is a reporting artifact compared against budget limits
    that themselves come from ``BudgetProfile`` (which is also ``float``),
    so promoting only this one record to ``Decimal`` would produce a
    spurious type mismatch without improving accuracy.

    Attributes:
        projected_cost: The pre-flight projected cost copied through from
            ``pre_flight.projected_cost``. Recorded in the result so the
            manifest entry is self-describing without needing to
            cross-reference ``PreFlightReport``.
        actual_gpu_hours: The elapsed GPU-hours observed during the run,
            as supplied by the caller.
        actual_cost: ``actual_gpu_hours * cost_rate_per_gpu_hour`` per
            Requirement 2.12 and design Property 7.
        absolute_diff: ``abs(actual_cost - projected_cost)`` per design
            Property 7. Always ``>= 0``.
        pct_diff: ``(absolute_diff / projected_cost) * 100`` when
            ``projected_cost > 0``, otherwise ``None`` (sentinel for the
            zero-projected-cost edge case).
        currency: Currency code propagated from
            ``pre_flight.projected_currency`` if present, else ``None``.
            Three-letter codes such as ``"USD"`` are typical but the
            ``BudgetProfile`` schema accepts any non-empty string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)

    # Subclasses override to point ``parse`` at the right error type. Not
    # used by ``reconcile_cost`` (which constructs the model directly) but
    # kept for parity with ``_DomainModel`` in ``math_lora.types.models``
    # in case a future caller wants to validate a serialized record from
    # disk.
    _schema_error_cls: ClassVar[type[SchemaValidationError]] = SchemaValidationError

    projected_cost: Annotated[StrictFloat | StrictInt, Field(ge=0.0)]
    actual_gpu_hours: Annotated[StrictFloat | StrictInt, Field(ge=0.0)]
    actual_cost: Annotated[StrictFloat | StrictInt, Field(ge=0.0)]
    absolute_diff: Annotated[StrictFloat | StrictInt, Field(ge=0.0)]
    # ``pct_diff`` may be ``None`` when ``projected_cost == 0`` (documented
    # sentinel). When present it is constrained to ``>= 0`` because
    # ``absolute_diff`` is already non-negative and the divisor is strictly
    # positive on this branch.
    pct_diff: Annotated[StrictFloat | StrictInt, Field(ge=0.0)] | None = None
    currency: StrictStr | None = None


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------


def reconcile_cost(
    pre_flight: PreFlightCostInputs,
    actual_gpu_hours: float,
) -> CostReconciliation:
    """Compute the end-of-run cost reconciliation (Requirement 2.12).

    The function is total, pure, and deterministic: it has no side effects,
    performs no I/O, and depends only on its two arguments.

    Args:
        pre_flight: Any object exposing ``projected_cost`` and
            ``cost_rate_per_gpu_hour`` attributes (see
            :class:`PreFlightCostInputs`). In production this is the
            :class:`PreFlightReport` produced by
            :func:`math_lora.planner.plan` (task 3.5); in tests a small
            structurally-compatible stub is sufficient.
        actual_gpu_hours: The elapsed GPU-hours observed during the run.
            Must be a non-negative real number; a value of ``0.0`` is
            permitted (e.g. a run that halted before the first training
            step) and produces ``actual_cost == 0.0``.

    Returns:
        A :class:`CostReconciliation` recording ``projected_cost``,
        ``actual_gpu_hours``, ``actual_cost``, ``absolute_diff``,
        ``pct_diff``, and ``currency``.

    Raises:
        ValueError: If ``actual_gpu_hours`` is negative or non-finite, or
            if the values exposed by ``pre_flight`` are negative or
            non-finite. These are programmer errors (the caller is expected
            to supply already-validated inputs) but we still guard against
            them so a stray ``nan`` cannot poison the manifest.

    Edge cases
    ----------

    * ``actual_gpu_hours == 0`` -> ``actual_cost == 0`` and
      ``absolute_diff == projected_cost``. ``pct_diff`` is ``None`` if
      ``projected_cost == 0`` (zero/zero is undefined; sentinel applies),
      otherwise ``100.0`` (the run consumed nothing relative to a non-zero
      projection).
    * ``projected_cost == 0`` -> ``pct_diff is None``  (documented
      sentinel; division by zero would otherwise raise).
    * ``actual_cost == projected_cost`` -> ``absolute_diff == 0`` and
      ``pct_diff == 0.0`` (when the projection was non-zero).

    Determinism
    -----------

    The function is a closed-form arithmetic expression over its inputs;
    repeated invocation with identical inputs returns identical outputs.
    This is the property tested by task 3.10 (Property 7).
    """

    # --- Stage 1: defensive input checks ---------------------------------
    # The function is pure and small, but a stray ``nan``/``inf`` in
    # ``actual_gpu_hours`` would silently propagate into the manifest and
    # break downstream JSON serialisation (NaN is not valid JSON). We
    # surface a clear ``ValueError`` instead.
    _check_finite_non_negative("actual_gpu_hours", float(actual_gpu_hours))
    projected_cost = float(pre_flight.projected_cost)
    cost_rate_per_gpu_hour = float(pre_flight.cost_rate_per_gpu_hour)
    _check_finite_non_negative("projected_cost", projected_cost)
    _check_finite_non_negative("cost_rate_per_gpu_hour", cost_rate_per_gpu_hour)

    # --- Stage 2: arithmetic --------------------------------------------
    # All three formulas come from design.md § Property 7. We compute
    # ``actual_cost`` first because both ``absolute_diff`` and
    # ``pct_diff`` depend on it.
    actual_cost = actual_gpu_hours * cost_rate_per_gpu_hour

    # ``abs(...)`` rather than the signed difference: design Property 7
    # explicitly specifies the absolute value. The signed direction can
    # always be recovered downstream from ``actual_cost`` vs.
    # ``projected_cost`` if a future report needs over/under-spend
    # signalling.
    absolute_diff = abs(actual_cost - projected_cost)

    # Zero-projected-cost sentinel. Division by zero is undefined; the
    # design's "when ``projected_cost > 0``" qualifier on Property 7
    # carves out exactly this case. Returning ``None`` (mapped to JSON
    # ``null`` by the manifest serialiser) is unambiguous and prevents
    # downstream code from confusing a sentinel with a real percentage.
    if projected_cost == 0.0:
        pct_diff: float | None = None
    else:
        pct_diff = (absolute_diff / projected_cost) * 100.0

    # --- Stage 3: optional currency propagation -------------------------
    # ``projected_currency`` is not part of the ``PreFlightCostInputs``
    # protocol (see protocol docstring rationale); we read it via
    # ``getattr`` so callers with minimal stubs are not forced to declare
    # it. Real ``PreFlightReport`` instances (task 3.5) will provide it.
    currency = getattr(pre_flight, "projected_currency", None)
    if currency is not None and not isinstance(currency, str):
        # Defensive: a non-string currency would be a programmer error in
        # the producer of ``pre_flight``. Surface it rather than embedding
        # a wrong-type value in the manifest.
        raise ValueError(
            "pre_flight.projected_currency must be a string when present, "
            f"got {type(currency).__name__}"
        )

    return CostReconciliation(
        projected_cost=projected_cost,
        actual_gpu_hours=actual_gpu_hours,
        actual_cost=actual_cost,
        absolute_diff=absolute_diff,
        pct_diff=pct_diff,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_finite_non_negative(name: str, value: float) -> None:
    """Reject ``nan``, ``inf``, and negative values for monetary inputs.

    Kept as a module-private helper so the three call sites in
    :func:`reconcile_cost` read uniformly and so a future addition (e.g.
    ``actual_cost`` self-check) can reuse it without duplication.
    """

    # ``nan``-aware: ``value < 0`` is False for ``nan``, so the explicit
    # ``value != value`` check (the canonical NaN test) catches it. We
    # could also use ``math.isnan`` / ``math.isfinite`` but staying in
    # pure Python avoids the import for what is a hot, tiny path.
    if value != value:  # NaN
        raise ValueError(f"{name} must not be NaN")
    if value == float("inf") or value == float("-inf"):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


__all__ = [
    "CostReconciliation",
    "PreFlightCostInputs",
    "reconcile_cost",
]
