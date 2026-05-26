"""Unit tests for :func:`math_lora.planner.reconcile_cost` (Requirement 2.12).

Coverage
--------

Task 3.9 of the implementation plan covers Requirement 2.12 of the spec:

    WHERE rented cloud GPUs are used, THE Training_Pipeline SHALL emit, at
    the end of the run, a cost reconciliation that records the projected
    cost from criterion 5, the actual elapsed GPU-hours, the actual cost
    computed as actual elapsed GPU-hours times the declared cost rate, and
    the absolute and percentage difference between projected and actual
    cost.

The arithmetic contract follows **design.md Property 7**:

* ``actual_cost == actual_gpu_hours * cost_rate_per_gpu_hour``
* ``absolute_diff == abs(actual_cost - projected_cost)``
* ``pct_diff == (absolute_diff / projected_cost) * 100`` when
  ``projected_cost > 0``; ``None`` (sentinel) when ``projected_cost == 0``.

Test cases below cover every scenario the task prompt enumerates -- typical
case, zero ``actual_gpu_hours``, equal projected and actual, the
``projected_cost == 0`` edge case, and field types/rounding -- plus a few
additional contract checks (input validation, currency propagation,
purity, frozen result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from math_lora.planner import (
    CostReconciliation,
    PreFlightCostInputs,
    reconcile_cost,
)


# ---------------------------------------------------------------------------
# Test fixtures: minimal pre_flight stub
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreFlightStub:
    """Smallest stub satisfying :class:`PreFlightCostInputs`.

    A frozen dataclass is used (rather than a real ``PreFlightReport``)
    because task 3.5 has not yet implemented that class. The function under
    test relies on ``Protocol`` duck-typing, so a stub with the two
    required attributes -- plus an optional ``projected_currency`` -- is
    sufficient and keeps the tests independent of upstream tasks.
    """

    projected_cost: float
    cost_rate_per_gpu_hour: float
    projected_currency: str | None = None


# ---------------------------------------------------------------------------
# Typical case
# ---------------------------------------------------------------------------


class TestReconcileCostTypicalCase:
    """Typical happy-path arithmetic over realistic cloud-GPU values."""

    @pytest.mark.unit
    def test_overspend_produces_positive_absolute_diff_and_pct(self) -> None:
        # Projection of $5.00 vs. actual 12 GPU-hours at $0.50/hr = $6.00.
        # Overspend by $1.00 == 20% relative to projection.
        pre_flight = _PreFlightStub(
            projected_cost=5.0,
            cost_rate_per_gpu_hour=0.5,
            projected_currency="USD",
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=12.0)

        assert result.actual_cost == pytest.approx(6.0)
        assert result.absolute_diff == pytest.approx(1.0)
        # ``absolute_diff / projected_cost * 100`` per design Property 7.
        assert result.pct_diff == pytest.approx(20.0)
        assert result.currency == "USD"
        # Inputs are echoed for manifest self-description.
        assert result.projected_cost == pytest.approx(5.0)
        assert result.actual_gpu_hours == pytest.approx(12.0)

    @pytest.mark.unit
    def test_underspend_produces_positive_absolute_diff_and_pct(self) -> None:
        # Projection of $10.00 vs. actual 6 GPU-hours at $1.00/hr = $6.00.
        # Underspend by $4.00 == 40% relative to projection. The function
        # returns ``abs(...)`` per design Property 7, so the magnitude is
        # the same regardless of direction.
        pre_flight = _PreFlightStub(
            projected_cost=10.0,
            cost_rate_per_gpu_hour=1.0,
            projected_currency="USD",
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=6.0)

        assert result.actual_cost == pytest.approx(6.0)
        assert result.absolute_diff == pytest.approx(4.0)
        assert result.pct_diff == pytest.approx(40.0)

    @pytest.mark.unit
    def test_realistic_h100_run(self) -> None:
        # Plausible production-scale numbers: 20 GPU-hours at $2.50/hr
        # vs. a $48.00 projection. Actual = $50.00, abs diff $2.00, ~4.17%.
        pre_flight = _PreFlightStub(
            projected_cost=48.0,
            cost_rate_per_gpu_hour=2.5,
            projected_currency="USD",
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=20.0)

        assert result.actual_cost == pytest.approx(50.0)
        assert result.absolute_diff == pytest.approx(2.0)
        assert result.pct_diff == pytest.approx(2.0 / 48.0 * 100.0)


# ---------------------------------------------------------------------------
# Zero ``actual_gpu_hours`` edge case
# ---------------------------------------------------------------------------


class TestReconcileCostZeroActualHours:
    """A run that halts before the first training step has zero GPU-hours."""

    @pytest.mark.unit
    def test_zero_actual_hours_with_nonzero_projection_yields_full_underspend(
        self,
    ) -> None:
        # actual_cost = 0 * rate = 0. abs(0 - 5) = 5. pct = 5/5*100 = 100%.
        pre_flight = _PreFlightStub(
            projected_cost=5.0, cost_rate_per_gpu_hour=0.5
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=0.0)

        assert result.actual_cost == pytest.approx(0.0)
        assert result.absolute_diff == pytest.approx(5.0)
        assert result.pct_diff == pytest.approx(100.0)

    @pytest.mark.unit
    def test_zero_actual_hours_with_zero_projection_yields_sentinel(self) -> None:
        # Both sides are zero: actual_cost == 0, abs_diff == 0, but
        # pct_diff is the documented ``None`` sentinel because
        # ``projected_cost == 0`` makes the percentage undefined.
        pre_flight = _PreFlightStub(
            projected_cost=0.0, cost_rate_per_gpu_hour=0.0
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=0.0)

        assert result.actual_cost == 0.0
        assert result.absolute_diff == 0.0
        assert result.pct_diff is None


# ---------------------------------------------------------------------------
# Equal projected and actual
# ---------------------------------------------------------------------------


class TestReconcileCostEqualProjectedAndActual:
    """When the projection matched reality, both diffs collapse to zero."""

    @pytest.mark.unit
    def test_projection_matches_actual_yields_zero_diffs(self) -> None:
        # 8 GPU-hours at $0.50/hr == $4.00, exactly the projection.
        pre_flight = _PreFlightStub(
            projected_cost=4.0, cost_rate_per_gpu_hour=0.5
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=8.0)

        assert result.actual_cost == pytest.approx(4.0)
        assert result.absolute_diff == pytest.approx(0.0)
        # 0/projected*100 == 0.0 (NOT None: the projection was non-zero
        # so the percentage is well-defined and equals zero).
        assert result.pct_diff == pytest.approx(0.0)
        assert result.pct_diff is not None


# ---------------------------------------------------------------------------
# ``projected_cost == 0`` edge case
# ---------------------------------------------------------------------------


class TestReconcileCostZeroProjectedCost:
    """The documented ``pct_diff = None`` sentinel for zero projections."""

    @pytest.mark.unit
    def test_zero_projected_cost_returns_none_pct_diff(self) -> None:
        # A run on locally-owned hardware (cost_rate == 0) but with a
        # non-zero actual-hours value. Both costs are zero so the absolute
        # diff is zero; pct is None per the sentinel.
        pre_flight = _PreFlightStub(
            projected_cost=0.0, cost_rate_per_gpu_hour=0.0
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=10.0)

        assert result.actual_cost == 0.0
        assert result.absolute_diff == 0.0
        assert result.pct_diff is None

    @pytest.mark.unit
    def test_zero_projected_cost_with_nonzero_rate_returns_none_pct_diff(
        self,
    ) -> None:
        # Stress case: the projection is zero (perhaps because batch_size
        # was set to 0 in a preflight what-if), but the cost rate is
        # non-zero so actual_cost > 0. ``absolute_diff`` is non-zero;
        # ``pct_diff`` is still None because the divisor is zero.
        pre_flight = _PreFlightStub(
            projected_cost=0.0, cost_rate_per_gpu_hour=0.5
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=2.0)

        assert result.actual_cost == pytest.approx(1.0)
        assert result.absolute_diff == pytest.approx(1.0)
        assert result.pct_diff is None


# ---------------------------------------------------------------------------
# Field types and structure
# ---------------------------------------------------------------------------


class TestReconcileCostResultStructure:
    """The result is a frozen :class:`CostReconciliation` with the right types."""

    @pytest.mark.unit
    def test_result_is_cost_reconciliation_instance(self) -> None:
        result = reconcile_cost(
            _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
            actual_gpu_hours=10.0,
        )
        assert isinstance(result, CostReconciliation)

    @pytest.mark.unit
    def test_arithmetic_fields_are_floats(self) -> None:
        result = reconcile_cost(
            _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
            actual_gpu_hours=10.0,
        )
        assert isinstance(result.projected_cost, float)
        assert isinstance(result.actual_gpu_hours, float)
        assert isinstance(result.actual_cost, float)
        assert isinstance(result.absolute_diff, float)
        assert isinstance(result.pct_diff, float)

    @pytest.mark.unit
    def test_pct_diff_is_none_or_float(self) -> None:
        # When pct_diff is computed, it's a real float; when sentinel-ed,
        # it's None. No other type ever appears.
        zero_proj = reconcile_cost(
            _PreFlightStub(projected_cost=0.0, cost_rate_per_gpu_hour=1.0),
            actual_gpu_hours=1.0,
        )
        assert zero_proj.pct_diff is None

        nonzero_proj = reconcile_cost(
            _PreFlightStub(projected_cost=2.0, cost_rate_per_gpu_hour=1.0),
            actual_gpu_hours=1.0,
        )
        assert isinstance(nonzero_proj.pct_diff, float)

    @pytest.mark.unit
    def test_result_is_frozen(self) -> None:
        # Manifest values must be immutable so downstream code cannot
        # mutate them after they have been recorded. Pydantic v2
        # enforces frozen by raising ``ValidationError`` on assignment.
        result = reconcile_cost(
            _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
            actual_gpu_hours=10.0,
        )
        with pytest.raises(Exception):  # ValidationError on frozen model
            result.actual_cost = 99.0  # type: ignore[misc]

    @pytest.mark.unit
    def test_currency_propagated_when_present(self) -> None:
        result = reconcile_cost(
            _PreFlightStub(
                projected_cost=1.0,
                cost_rate_per_gpu_hour=0.1,
                projected_currency="EUR",
            ),
            actual_gpu_hours=10.0,
        )
        assert result.currency == "EUR"

    @pytest.mark.unit
    def test_currency_absent_yields_none(self) -> None:
        # Bare protocol implementer with no ``projected_currency`` attr.
        @dataclass(frozen=True)
        class _Bare:
            projected_cost: float
            cost_rate_per_gpu_hour: float

        result = reconcile_cost(
            _Bare(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
            actual_gpu_hours=10.0,
        )
        assert result.currency is None


# ---------------------------------------------------------------------------
# Rounding / floating-point precision
# ---------------------------------------------------------------------------


class TestReconcileCostRounding:
    """Floating-point precision at typical cloud-GPU magnitudes."""

    @pytest.mark.unit
    def test_no_premature_rounding(self) -> None:
        # Values that would round-trip cleanly only at full float64
        # precision. The function does not round (rounding is a
        # presentation concern, not a reconciliation concern), so the
        # output should match the unrounded arithmetic exactly.
        pre_flight = _PreFlightStub(
            projected_cost=0.123456789, cost_rate_per_gpu_hour=0.0123
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=7.89)

        expected_actual = 7.89 * 0.0123
        expected_abs = abs(expected_actual - 0.123456789)
        expected_pct = expected_abs / 0.123456789 * 100.0

        assert result.actual_cost == pytest.approx(expected_actual)
        assert result.absolute_diff == pytest.approx(expected_abs)
        assert result.pct_diff == pytest.approx(expected_pct)

    @pytest.mark.unit
    def test_pct_diff_can_exceed_one_hundred(self) -> None:
        # The function does not clamp the percentage; a 5x overspend
        # yields a 400% pct_diff, which is the correct value to record
        # even though it exceeds 100. This is a regression check
        # against any future "looks like a probability, must be in
        # [0, 100]" clamping.
        pre_flight = _PreFlightStub(
            projected_cost=1.0, cost_rate_per_gpu_hour=1.0
        )
        result = reconcile_cost(pre_flight, actual_gpu_hours=5.0)

        assert result.actual_cost == pytest.approx(5.0)
        assert result.absolute_diff == pytest.approx(4.0)
        assert result.pct_diff == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Determinism / purity
# ---------------------------------------------------------------------------


class TestReconcileCostPurity:
    """Repeated calls with identical inputs return identical outputs."""

    @pytest.mark.unit
    def test_repeated_call_returns_equal_result(self) -> None:
        pre_flight = _PreFlightStub(
            projected_cost=10.0,
            cost_rate_per_gpu_hour=0.5,
            projected_currency="USD",
        )
        first = reconcile_cost(pre_flight, actual_gpu_hours=18.0)
        second = reconcile_cost(pre_flight, actual_gpu_hours=18.0)
        # Pydantic v2 frozen models compare by field equality.
        assert first == second


# ---------------------------------------------------------------------------
# Input validation: defensive guards against NaN / inf / negative
# ---------------------------------------------------------------------------


class TestReconcileCostInputValidation:
    """``ValueError`` on programmer errors that would corrupt the manifest."""

    @pytest.mark.unit
    def test_negative_actual_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="actual_gpu_hours.*non-negative"):
            reconcile_cost(
                _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
                actual_gpu_hours=-0.001,
            )

    @pytest.mark.unit
    def test_nan_actual_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            reconcile_cost(
                _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
                actual_gpu_hours=float("nan"),
            )

    @pytest.mark.unit
    def test_inf_actual_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            reconcile_cost(
                _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1),
                actual_gpu_hours=float("inf"),
            )

    @pytest.mark.unit
    def test_negative_projected_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="projected_cost.*non-negative"):
            reconcile_cost(
                _PreFlightStub(
                    projected_cost=-1.0, cost_rate_per_gpu_hour=0.1
                ),
                actual_gpu_hours=1.0,
            )

    @pytest.mark.unit
    def test_negative_cost_rate_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="cost_rate_per_gpu_hour.*non-negative"
        ):
            reconcile_cost(
                _PreFlightStub(
                    projected_cost=1.0, cost_rate_per_gpu_hour=-0.5
                ),
                actual_gpu_hours=1.0,
            )

    @pytest.mark.unit
    def test_non_string_currency_rejected(self) -> None:
        # Programmer error: ``projected_currency`` should be a string per
        # the BudgetProfile schema. A non-string value here would slip
        # into the manifest and break JSON serialisation downstream.
        @dataclass(frozen=True)
        class _BadCurrency:
            projected_cost: float
            cost_rate_per_gpu_hour: float
            projected_currency: Any

        with pytest.raises(ValueError, match="currency"):
            reconcile_cost(
                _BadCurrency(
                    projected_cost=1.0,
                    cost_rate_per_gpu_hour=0.1,
                    projected_currency=123,
                ),
                actual_gpu_hours=1.0,
            )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestPreFlightCostInputsProtocol:
    """The :class:`PreFlightCostInputs` protocol matches minimal stubs."""

    @pytest.mark.unit
    def test_stub_satisfies_protocol_at_runtime(self) -> None:
        # ``PreFlightCostInputs`` is ``@runtime_checkable`` so callers can
        # do ``isinstance`` if they want. We do not rely on that in the
        # function itself (duck typing is sufficient) but the tests
        # exercise it as a contract.
        stub = _PreFlightStub(projected_cost=1.0, cost_rate_per_gpu_hour=0.1)
        assert isinstance(stub, PreFlightCostInputs)
