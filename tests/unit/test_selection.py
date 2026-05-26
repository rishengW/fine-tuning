"""Unit tests for the eligibility, feasibility, and scoring primitives (task 2.3).

These are example-based tests that complement the property-based tests in
task 2.5 (selection-report soundness, Property 2). Together they cover the
acceptance criteria called out by task 2.3:

* Requirement 1.1 -- declared input fields per candidate
* Requirement 1.4 -- feasibility flag and ``vram_shortfall_gb`` rounded to
  one decimal place
* Requirement 1.5 -- weighted scoring formula and lexicographic tie-breaker
* Requirement 1.8 -- license permissiveness flags
* Requirement 1.10 -- ineligibility for missing fields, with the absent
  field names recorded

Each test class focuses on one primitive (eligibility, feasibility,
scoring, tie-break, end-to-end candidate evaluation). Test names spell out
the law being checked so a regression fails with a readable error.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from math_lora.model_selector import (
    REQUIRED_FIELDS,
    CandidateEntry,
    EligibilityResult,
    FeasibilityResult,
    ScoredCandidate,
    ScoringInputs,
    ScoringWeights,
    check_eligibility,
    check_feasibility,
    compute_score,
    evaluate_candidate,
    license_permissiveness,
    pick_winner,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _valid_candidate_payload(
    *, model_id: str = "qwen/Qwen2.5-Math-7B"
) -> dict[str, Any]:
    """Return a fully-populated raw candidate payload.

    Tests mutate one field at a time on a copy of this dict so that every
    rejection isolates a single violation.
    """

    return {
        "model_id": model_id,
        "revision": "main",
        "family": "qwen",
        "param_count_b": 7.0,
        "license_id": "Apache-2.0",
        "license_allows_finetuning": True,
        "license_allows_adapter_redistribution": True,
        "license_allows_commercial_use": True,
        "native_context_length_tokens": 4096,
        "tokenizer_family": "qwen",
        "baseline_gsm8k": 0.85,
        "baseline_math": 0.55,
    }


# ---------------------------------------------------------------------------
# Eligibility (Requirement 1.10)
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Sanity checks on the :data:`REQUIRED_FIELDS` table.

    These guard the Req 1.1 / 1.8 contract at the table level so a future
    refactor cannot silently drop a required field name.
    """

    @pytest.mark.unit
    def test_required_fields_table_is_deduplicated(self) -> None:
        # ``license_id`` appears in both Req 1.1 and Req 1.8 but must
        # appear only once in the eligibility report -- otherwise the
        # ``missing_fields`` list would name it twice.
        assert len(REQUIRED_FIELDS) == len(set(REQUIRED_FIELDS))

    @pytest.mark.unit
    def test_required_fields_table_covers_req_1_1_and_1_8(self) -> None:
        # Spell out the union here so a future refactor that drops a name
        # from the table fails this test loudly.
        expected = {
            # Req 1.1
            "model_id",
            "param_count_b",
            "license_id",
            "native_context_length_tokens",
            "tokenizer_family",
            "baseline_gsm8k",
            "baseline_math",
            # Req 1.8
            "license_allows_finetuning",
            "license_allows_adapter_redistribution",
            "license_allows_commercial_use",
        }
        assert set(REQUIRED_FIELDS) == expected


class TestCheckEligibility:
    """Tests for :func:`check_eligibility` (Requirement 1.10)."""

    @pytest.mark.unit
    def test_fully_populated_payload_is_eligible(self) -> None:
        result = check_eligibility(_valid_candidate_payload())

        assert result.eligible is True
        assert result.missing_fields == ()

    @pytest.mark.unit
    @pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
    def test_missing_single_field_is_ineligible_with_field_named(
        self, missing_field: str
    ) -> None:
        # Drop one required field at a time. The result must be
        # ineligible, and the dropped field name must appear in the
        # surfaced ``missing_fields`` tuple per Requirement 1.10.
        payload = _valid_candidate_payload()
        del payload[missing_field]

        result = check_eligibility(payload)

        assert result.eligible is False
        assert missing_field in result.missing_fields

    @pytest.mark.unit
    @pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
    def test_none_valued_field_is_treated_as_absent(
        self, missing_field: str
    ) -> None:
        # YAML/JSON loaders frequently surface missing-but-declared keys
        # as ``None``. Eligibility treats ``None`` the same as a missing
        # key so the operator gets a uniform error.
        payload = _valid_candidate_payload()
        payload[missing_field] = None

        result = check_eligibility(payload)

        assert result.eligible is False
        assert missing_field in result.missing_fields

    @pytest.mark.unit
    def test_multiple_missing_fields_are_all_named(self) -> None:
        # When several fields are missing, every one of them must appear
        # in the result -- the operator should fix all problems at once,
        # not one per round-trip through the selector.
        payload = _valid_candidate_payload()
        del payload["param_count_b"]
        del payload["license_allows_commercial_use"]
        del payload["baseline_gsm8k"]

        result = check_eligibility(payload)

        assert result.eligible is False
        for name in (
            "param_count_b",
            "license_allows_commercial_use",
            "baseline_gsm8k",
        ):
            assert name in result.missing_fields

    @pytest.mark.unit
    def test_missing_fields_preserves_declaration_order(self) -> None:
        # The tuple must appear in the order the names occur in
        # :data:`REQUIRED_FIELDS`. That ordering is the order of the
        # requirements document, so the operator's report reads top-to-
        # bottom in the same sequence as the spec.
        result = check_eligibility({})  # everything missing

        assert result.missing_fields == REQUIRED_FIELDS

    @pytest.mark.unit
    def test_extra_fields_are_ignored_for_eligibility(self) -> None:
        # Eligibility only checks presence of *required* fields. Extra
        # fields (e.g., a vendor-specific annotation) do not affect the
        # verdict here -- the schema layer handles strictness via
        # ``extra="forbid"`` on the pydantic model.
        payload = _valid_candidate_payload()
        payload["vendor_note"] = "internal preview"

        result = check_eligibility(payload)

        assert result.eligible is True

    @pytest.mark.unit
    def test_empty_string_values_are_present_for_eligibility(self) -> None:
        # Eligibility splits "absent" from "invalid". An empty string is
        # invalid (the schema layer rejects it) but it is *present* -- so
        # the eligibility check returns ``eligible=True`` and the schema
        # layer is responsible for the value rejection. This split is
        # what lets the selector report the candidate at all instead of
        # raising during eligibility.
        payload = _valid_candidate_payload()
        payload["license_id"] = ""

        result = check_eligibility(payload)

        assert result.eligible is True
        assert result.missing_fields == ()


# ---------------------------------------------------------------------------
# Feasibility (Requirement 1.4)
# ---------------------------------------------------------------------------


class TestCheckFeasibility:
    """Tests for :func:`check_feasibility` (Requirement 1.4)."""

    @pytest.mark.unit
    def test_estimated_below_available_is_feasible(self) -> None:
        result = check_feasibility(estimated_min_vram_gb=10.0, available_vram_gb=16.0)

        assert result.feasible is True
        assert result.vram_shortfall_gb is None

    @pytest.mark.unit
    def test_estimated_equal_to_available_is_feasible(self) -> None:
        # Property 2(b) pins ``feasible == (estimated <= available)``.
        # The boundary case ``estimated == available`` is feasible.
        result = check_feasibility(estimated_min_vram_gb=24.0, available_vram_gb=24.0)

        assert result.feasible is True
        assert result.vram_shortfall_gb is None

    @pytest.mark.unit
    def test_estimated_above_available_is_infeasible(self) -> None:
        result = check_feasibility(estimated_min_vram_gb=30.0, available_vram_gb=24.0)

        assert result.feasible is False
        # Property 2(c): shortfall == round(estimated - available, 1).
        assert result.vram_shortfall_gb == 6.0

    @pytest.mark.unit
    def test_shortfall_is_rounded_to_one_decimal_place(self) -> None:
        # Requirement 1.4: "SHALL record the shortfall in gigabytes
        # rounded to one decimal place". A raw difference of 5.6789 must
        # be reported as 5.7, not 5.7 plus noise.
        result = check_feasibility(
            estimated_min_vram_gb=29.6789, available_vram_gb=24.0
        )

        assert result.feasible is False
        assert result.vram_shortfall_gb == round(29.6789 - 24.0, 1)
        assert result.vram_shortfall_gb == 5.7

    @pytest.mark.unit
    def test_shortfall_handles_floats_with_long_decimals(self) -> None:
        # A real-world VRAM estimate (from the activation proxy) easily
        # produces 8+ decimal digits. The reported shortfall must collapse
        # to the documented one-decimal-place form.
        estimated = 24.135792468
        available = 24.0
        result = check_feasibility(
            estimated_min_vram_gb=estimated, available_vram_gb=available
        )

        assert result.feasible is False
        # The rounded value: 24.135792468 - 24.0 = 0.135792468 -> 0.1.
        assert result.vram_shortfall_gb == 0.1

    @pytest.mark.unit
    def test_strictly_positive_shortfall_when_infeasible(self) -> None:
        # An infeasible candidate cannot have ``shortfall == 0`` by
        # construction: ``estimated <= available`` would have made it
        # feasible. The dataclass invariant therefore is ``shortfall > 0``
        # whenever ``feasible is False``.
        result = check_feasibility(
            estimated_min_vram_gb=24.05, available_vram_gb=24.0
        )

        assert result.feasible is False
        assert result.vram_shortfall_gb is not None
        assert result.vram_shortfall_gb > 0.0


# ---------------------------------------------------------------------------
# License permissiveness (Requirement 1.5 / 1.8)
# ---------------------------------------------------------------------------


class TestLicensePermissiveness:
    """Tests for :func:`license_permissiveness`."""

    @pytest.mark.unit
    def test_all_three_flags_true_is_one(self) -> None:
        # Fully-permissive license: maximum permissiveness.
        assert license_permissiveness(True, True, True) == 1.0

    @pytest.mark.unit
    def test_all_three_flags_false_is_zero(self) -> None:
        # Fully-restrictive license: minimum permissiveness.
        assert license_permissiveness(False, False, False) == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("ft", "redist", "commercial", "expected"),
        [
            (True, False, False, 1.0 / 3.0),
            (False, True, False, 1.0 / 3.0),
            (False, False, True, 1.0 / 3.0),
            (True, True, False, 2.0 / 3.0),
            (True, False, True, 2.0 / 3.0),
            (False, True, True, 2.0 / 3.0),
        ],
    )
    def test_mixed_flags_produce_thirds(
        self, ft: bool, redist: bool, commercial: bool, expected: float
    ) -> None:
        # Mean of three booleans: each True flag contributes 1/3.
        assert math.isclose(
            license_permissiveness(ft, redist, commercial),
            expected,
            rel_tol=1e-12,
            abs_tol=0.0,
        )

    @pytest.mark.unit
    def test_value_is_in_unit_interval(self) -> None:
        # Property 2(f) requires every score-input in [0, 1]; the
        # permissiveness value feeds into the score directly so it must
        # also live in that interval.
        for flags in [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ]:
            value = license_permissiveness(*flags)
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Scoring weights (Requirement 1.5)
# ---------------------------------------------------------------------------


class TestScoringWeights:
    """Constructor-level tests for :class:`ScoringWeights`."""

    @pytest.mark.unit
    def test_default_weights_sum_to_one(self) -> None:
        # Requirement 1.5: weights sum to 1.0.
        w = ScoringWeights()

        total = w.w_gsm + w.w_math + w.w_params + w.w_license
        assert math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=0.0)

    @pytest.mark.unit
    def test_custom_weights_summing_to_one_are_accepted(self) -> None:
        # Operator may override the default split, e.g. to weight MATH
        # more heavily than GSM8K.
        w = ScoringWeights(w_gsm=0.2, w_math=0.5, w_params=0.1, w_license=0.2)

        total = w.w_gsm + w.w_math + w.w_params + w.w_license
        assert math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=0.0)

    @pytest.mark.unit
    def test_weights_summing_above_one_are_rejected(self) -> None:
        # 0.3 + 0.4 + 0.2 + 0.2 = 1.1 -- a typo a real operator might
        # make. Must be rejected so the score stays in [0, 1].
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(w_gsm=0.3, w_math=0.4, w_params=0.2, w_license=0.2)

    @pytest.mark.unit
    def test_weights_summing_below_one_are_rejected(self) -> None:
        # 0.3 + 0.3 + 0.2 + 0.1 = 0.9 -- another easy typo.
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(w_gsm=0.3, w_math=0.3, w_params=0.2, w_license=0.1)

    @pytest.mark.unit
    def test_negative_weight_is_rejected(self) -> None:
        # A negative weight would invert the contribution sign and can
        # produce a negative score; banned by the constructor invariant.
        with pytest.raises(ValueError, match="must be in"):
            ScoringWeights(
                w_gsm=-0.1, w_math=0.4, w_params=0.3, w_license=0.4
            )

    @pytest.mark.unit
    def test_weight_above_one_is_rejected(self) -> None:
        # Catches clearly out-of-range typos even when the sum happens to
        # be 1.0 (e.g., one weight is 1.5 and another is -0.5).
        with pytest.raises(ValueError, match="must be in"):
            ScoringWeights(
                w_gsm=1.5, w_math=-0.5, w_params=0.0, w_license=0.0
            )


# ---------------------------------------------------------------------------
# compute_score (Requirement 1.5)
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Tests for :func:`compute_score` (Requirement 1.5)."""

    @pytest.mark.unit
    def test_score_lies_in_unit_interval(self) -> None:
        # Property 2(f): every score lies in [0.0, 1.0]. We check the
        # boundary cases here; the property test covers arbitrary inputs.
        worst = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.0,
                baseline_math=0.0,
                param_count_b=0.0,  # below the normalize lo bound
                license_permissiveness=0.0,
            )
        )
        best = compute_score(
            ScoringInputs(
                baseline_gsm8k=1.0,
                baseline_math=1.0,
                param_count_b=1000.0,  # well above the normalize hi bound
                license_permissiveness=1.0,
            )
        )

        assert worst == 0.0
        assert best == 1.0

    @pytest.mark.unit
    def test_score_is_deterministic(self) -> None:
        # Same inputs -> same output, byte-for-byte. Required by the
        # selection-report soundness property and by the bit-for-bit
        # reproducibility guarantee in Requirement 8.10.
        inputs = ScoringInputs(
            baseline_gsm8k=0.85,
            baseline_math=0.55,
            param_count_b=7.0,
            license_permissiveness=1.0,
        )

        a = compute_score(inputs)
        b = compute_score(inputs)

        assert a == b

    @pytest.mark.unit
    def test_score_matches_weighted_combination(self) -> None:
        # Spelling out the formula here pins the contract: a future
        # refactor that, e.g., normalizes baseline scores differently
        # will fail this exact-equality check.
        inputs = ScoringInputs(
            baseline_gsm8k=0.85,
            baseline_math=0.55,
            param_count_b=7.0,
            license_permissiveness=2.0 / 3.0,
        )
        weights = ScoringWeights()

        # Param normalization is linear in [0.5B, 70B] -> [0, 1].
        normalized_params = (7.0 - 0.5) / (70.0 - 0.5)
        expected = (
            weights.w_gsm * 0.85
            + weights.w_math * 0.55
            + weights.w_params * normalized_params
            + weights.w_license * (2.0 / 3.0)
        )

        result = compute_score(inputs, weights=weights)
        assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=0.0)

    @pytest.mark.unit
    def test_higher_gsm_baseline_yields_higher_score(self) -> None:
        # Monotonicity in the GSM8K input -- spot check.
        low = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.20,
                baseline_math=0.50,
                param_count_b=7.0,
                license_permissiveness=0.5,
            )
        )
        high = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.80,
                baseline_math=0.50,
                param_count_b=7.0,
                license_permissiveness=0.5,
            )
        )

        assert high > low

    @pytest.mark.unit
    def test_param_count_below_band_clamps_to_zero(self) -> None:
        # Param normalization clamps below ``_PARAMS_NORMALIZE_LO_B``;
        # two candidates that differ only in a sub-0.5B param count must
        # produce identical scores.
        below = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.5,
                baseline_math=0.5,
                param_count_b=0.1,
                license_permissiveness=0.5,
            )
        )
        at_floor = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.5,
                baseline_math=0.5,
                param_count_b=0.5,
                license_permissiveness=0.5,
            )
        )

        assert below == at_floor

    @pytest.mark.unit
    def test_param_count_above_band_clamps_to_one(self) -> None:
        # Symmetric to the previous test for the upper bound.
        at_ceiling = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.5,
                baseline_math=0.5,
                param_count_b=70.0,
                license_permissiveness=0.5,
            )
        )
        above = compute_score(
            ScoringInputs(
                baseline_gsm8k=0.5,
                baseline_math=0.5,
                param_count_b=405.0,
                license_permissiveness=0.5,
            )
        )

        assert above == at_ceiling


# ---------------------------------------------------------------------------
# pick_winner: lexicographic tie-breaker (Requirement 1.5)
# ---------------------------------------------------------------------------


class TestPickWinner:
    """Tests for :func:`pick_winner` -- ranking and tie-breaking."""

    @pytest.mark.unit
    def test_empty_input_returns_none(self) -> None:
        assert pick_winner([]) is None

    @pytest.mark.unit
    def test_single_candidate_is_the_winner(self) -> None:
        single = ScoredCandidate(model_id="qwen/A", score=0.42)

        assert pick_winner([single]) == single

    @pytest.mark.unit
    def test_higher_score_wins(self) -> None:
        a = ScoredCandidate(model_id="qwen/A", score=0.42)
        b = ScoredCandidate(model_id="qwen/B", score=0.71)
        c = ScoredCandidate(model_id="qwen/C", score=0.55)

        winner = pick_winner([a, b, c])

        assert winner == b

    @pytest.mark.unit
    def test_tie_resolves_by_lexicographic_smallest_model_id(self) -> None:
        # Three candidates, all at score 0.5. The lex-smallest model_id
        # ("aaa/A") wins per the documented tie-breaker.
        a = ScoredCandidate(model_id="qwen/A", score=0.5)
        b = ScoredCandidate(model_id="aaa/A", score=0.5)
        c = ScoredCandidate(model_id="zzz/A", score=0.5)

        winner = pick_winner([a, b, c])

        assert winner == b

    @pytest.mark.unit
    def test_tie_break_is_independent_of_input_order(self) -> None:
        # Two candidates at the same score, presented in three different
        # orders. The winner must be the same in every order.
        x = ScoredCandidate(model_id="alpha/X", score=0.7)
        y = ScoredCandidate(model_id="beta/Y", score=0.7)

        for order in [(x, y), (y, x)]:
            assert pick_winner(list(order)) == x

    @pytest.mark.unit
    def test_higher_score_beats_lex_smaller_id(self) -> None:
        # Tie-break is *only* used to resolve ties. A higher-scoring
        # candidate wins over a lex-smaller-id but lower-scoring one.
        worse_but_first = ScoredCandidate(model_id="aaa/A", score=0.10)
        better_but_last = ScoredCandidate(model_id="zzz/Z", score=0.99)

        assert pick_winner([worse_but_first, better_but_last]) == better_but_last

    @pytest.mark.unit
    def test_tie_break_is_case_sensitive(self) -> None:
        # Lex order is the natural Python string comparison; uppercase
        # letters sort before lowercase. Pinning this explicitly so a
        # future refactor that case-folds the model_id breaks loudly.
        upper = ScoredCandidate(model_id="QWEN/A", score=0.5)
        lower = ScoredCandidate(model_id="qwen/A", score=0.5)

        assert pick_winner([upper, lower]) == upper


# ---------------------------------------------------------------------------
# evaluate_candidate end-to-end (combines all three primitives)
# ---------------------------------------------------------------------------


class TestEvaluateCandidate:
    """End-to-end tests for :func:`evaluate_candidate`."""

    @pytest.mark.unit
    def test_eligible_feasible_candidate_has_score_and_no_shortfall(self) -> None:
        # Happy path: every required field present, VRAM fits.
        entry = evaluate_candidate(
            payload=_valid_candidate_payload(),
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )

        assert isinstance(entry, CandidateEntry)
        assert entry.eligible is True
        assert entry.feasible is True
        assert entry.vram_shortfall_gb is None
        assert entry.score is not None
        assert 0.0 <= entry.score <= 1.0
        assert entry.missing_fields == ()
        assert entry.declared_param_count_b == 7.0

    @pytest.mark.unit
    def test_eligible_infeasible_candidate_has_shortfall_and_no_score(self) -> None:
        # VRAM estimate exceeds the operator's available VRAM. Feasibility
        # is False, the shortfall is named (rounded to 1 dp), and the
        # candidate is excluded from ranking (score is None).
        entry = evaluate_candidate(
            payload=_valid_candidate_payload(),
            estimated_min_vram_gb=30.7,
            available_vram_gb=24.0,
        )

        assert entry.eligible is True
        assert entry.feasible is False
        assert entry.vram_shortfall_gb == 6.7
        assert entry.score is None

    @pytest.mark.unit
    def test_ineligible_candidate_has_no_score_and_lists_missing_fields(self) -> None:
        # Per Requirement 1.10: ineligible candidates are excluded from
        # ranking AND the absent field names are listed.
        payload = _valid_candidate_payload()
        del payload["baseline_gsm8k"]
        del payload["license_allows_commercial_use"]

        entry = evaluate_candidate(
            payload=payload,
            estimated_min_vram_gb=None,  # not estimated for ineligible
            available_vram_gb=24.0,
        )

        assert entry.eligible is False
        assert entry.feasible is False  # not evaluated -> not feasible
        assert entry.score is None
        assert entry.vram_shortfall_gb is None
        assert "baseline_gsm8k" in entry.missing_fields
        assert "license_allows_commercial_use" in entry.missing_fields

    @pytest.mark.unit
    def test_ineligible_candidate_still_reports_model_id_when_present(self) -> None:
        # Requirement 1.10 requires that the report list ineligible
        # candidates -- which means the row must be keyed by model_id
        # even when other fields are missing.
        payload = _valid_candidate_payload(model_id="vendor/Specialcase-7B")
        del payload["baseline_math"]

        entry = evaluate_candidate(
            payload=payload,
            estimated_min_vram_gb=None,
            available_vram_gb=24.0,
        )

        assert entry.model_id == "vendor/Specialcase-7B"
        assert entry.eligible is False

    @pytest.mark.unit
    def test_ineligible_candidate_with_missing_model_id_uses_empty_string(
        self,
    ) -> None:
        # Defensive: when even the model_id is missing, the row still has
        # to be total so the report does not crash. The placeholder is
        # the empty string; the operator can still see the ineligibility
        # cause via missing_fields.
        payload = _valid_candidate_payload()
        del payload["model_id"]

        entry = evaluate_candidate(
            payload=payload,
            estimated_min_vram_gb=None,
            available_vram_gb=24.0,
        )

        assert entry.eligible is False
        assert entry.model_id == ""
        assert "model_id" in entry.missing_fields

    @pytest.mark.unit
    def test_eligible_candidate_requires_vram_estimate(self) -> None:
        # Programming-error guard: the orchestrator must always supply a
        # VRAM estimate for an eligible candidate (otherwise feasibility
        # cannot be computed). The function surfaces this loudly.
        with pytest.raises(ValueError, match="estimated_min_vram_gb is required"):
            evaluate_candidate(
                payload=_valid_candidate_payload(),
                estimated_min_vram_gb=None,
                available_vram_gb=24.0,
            )

    @pytest.mark.unit
    def test_score_reflects_higher_baselines(self) -> None:
        # Two otherwise-identical eligible feasible candidates differing
        # only in their math baselines: the one with stronger baselines
        # must score higher.
        weak_payload = _valid_candidate_payload(model_id="weak/M-7B")
        weak_payload["baseline_gsm8k"] = 0.10
        weak_payload["baseline_math"] = 0.10

        strong_payload = _valid_candidate_payload(model_id="strong/M-7B")
        strong_payload["baseline_gsm8k"] = 0.90
        strong_payload["baseline_math"] = 0.90

        weak_entry = evaluate_candidate(
            payload=weak_payload,
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )
        strong_entry = evaluate_candidate(
            payload=strong_payload,
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )

        assert weak_entry.score is not None
        assert strong_entry.score is not None
        assert strong_entry.score > weak_entry.score

    @pytest.mark.unit
    def test_score_reflects_license_permissiveness(self) -> None:
        # Same baselines and size, different license flags. The fully-
        # permissive candidate must score strictly higher.
        restrictive = _valid_candidate_payload(model_id="closed/M-7B")
        restrictive["license_allows_finetuning"] = False
        restrictive["license_allows_adapter_redistribution"] = False
        restrictive["license_allows_commercial_use"] = False

        permissive = _valid_candidate_payload(model_id="open/M-7B")
        permissive["license_allows_finetuning"] = True
        permissive["license_allows_adapter_redistribution"] = True
        permissive["license_allows_commercial_use"] = True

        r_entry = evaluate_candidate(
            payload=restrictive,
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )
        p_entry = evaluate_candidate(
            payload=permissive,
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )

        assert r_entry.score is not None
        assert p_entry.score is not None
        assert p_entry.score > r_entry.score


# ---------------------------------------------------------------------------
# End-to-end tie-break across evaluate_candidate + pick_winner
# ---------------------------------------------------------------------------


class TestEvaluateAndPickIntegration:
    """The two primitives compose to satisfy Property 2(e)."""

    @pytest.mark.unit
    def test_winner_under_tie_is_lex_smallest_eligible_feasible(self) -> None:
        # Three identical-scoring candidates, ranked by name. The
        # lex-smallest model_id among the eligible feasible set wins
        # even when its insertion order is not first.
        common = {
            "revision": "main",
            "family": "qwen",
            "param_count_b": 7.0,
            "license_id": "Apache-2.0",
            "license_allows_finetuning": True,
            "license_allows_adapter_redistribution": True,
            "license_allows_commercial_use": True,
            "native_context_length_tokens": 4096,
            "tokenizer_family": "qwen",
            "baseline_gsm8k": 0.5,
            "baseline_math": 0.5,
        }
        payloads = [
            {"model_id": "zzz/M-7B", **common},
            {"model_id": "aaa/M-7B", **common},
            {"model_id": "mmm/M-7B", **common},
        ]
        scored: list[ScoredCandidate] = []
        for p in payloads:
            entry = evaluate_candidate(
                payload=p,
                estimated_min_vram_gb=14.0,
                available_vram_gb=24.0,
            )
            assert entry.score is not None
            scored.append(ScoredCandidate(model_id=entry.model_id, score=entry.score))

        # All three should have identical scores (same baselines + same
        # license + same size + same band normalization).
        scores = {sc.score for sc in scored}
        assert len(scores) == 1

        winner = pick_winner(scored)
        assert winner is not None
        assert winner.model_id == "aaa/M-7B"

    @pytest.mark.unit
    def test_ineligible_and_infeasible_candidates_are_excluded_from_ranking(
        self,
    ) -> None:
        # Three candidates: one ineligible (missing baseline_gsm8k),
        # one infeasible (VRAM estimate > available), one eligible
        # feasible. The winner is the third.
        ineligible_payload = _valid_candidate_payload(model_id="bad/Missing-7B")
        del ineligible_payload["baseline_gsm8k"]
        infeasible_payload = _valid_candidate_payload(model_id="big/Huge-70B")
        infeasible_payload["param_count_b"] = 70.0
        feasible_payload = _valid_candidate_payload(model_id="ok/Fits-7B")

        ineligible = evaluate_candidate(
            payload=ineligible_payload,
            estimated_min_vram_gb=None,
            available_vram_gb=24.0,
        )
        infeasible = evaluate_candidate(
            payload=infeasible_payload,
            estimated_min_vram_gb=140.0,
            available_vram_gb=24.0,
        )
        feasible = evaluate_candidate(
            payload=feasible_payload,
            estimated_min_vram_gb=14.0,
            available_vram_gb=24.0,
        )

        # Each non-rankable entry has score=None per Property 2.
        assert ineligible.score is None
        assert infeasible.score is None
        assert feasible.score is not None

        # The ranking is over the rankable subset only.
        rankable = [
            ScoredCandidate(model_id=e.model_id, score=e.score)
            for e in (ineligible, infeasible, feasible)
            if e.score is not None
        ]
        winner = pick_winner(rankable)

        assert winner is not None
        assert winner.model_id == "ok/Fits-7B"


# ---------------------------------------------------------------------------
# Determinism (used everywhere; called out explicitly)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Repeated invocations with identical inputs return identical outputs."""

    @pytest.mark.unit
    def test_evaluate_candidate_is_deterministic(self) -> None:
        payload = _valid_candidate_payload()

        a = evaluate_candidate(payload, 14.0, 24.0)
        b = evaluate_candidate(payload, 14.0, 24.0)

        assert a == b
        # ``frozen=True`` on the dataclass means ``a == b`` already implies
        # field-by-field equality. The line below documents that.
        assert isinstance(a, CandidateEntry) and isinstance(b, CandidateEntry)
        assert a.score == b.score

    @pytest.mark.unit
    def test_check_eligibility_returns_eligibility_result(self) -> None:
        # Smoke: the public API returns the documented dataclass, not a
        # bare bool or tuple.
        result = check_eligibility({})
        assert isinstance(result, EligibilityResult)

    @pytest.mark.unit
    def test_check_feasibility_returns_feasibility_result(self) -> None:
        result = check_feasibility(10.0, 24.0)
        assert isinstance(result, FeasibilityResult)
