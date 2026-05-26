"""Unit tests for ``Dataset_Builder`` canonicalization and deduplication (task 4.6).

Coverage matrix
---------------

Per task 4.6 of ``.kiro/specs/math-lora-finetuning/tasks.md``:

* Lowercasing (``Find X`` -> ``find x``).
* Trim leading/trailing whitespace.
* Internal whitespace collapse (single space, tab, newline, mixed runs).
* Trailing-punctuation stripping for **each** documented character and
  combinations.
* Idempotence: ``canonicalize(canonicalize(x)) == canonicalize(x)``.
* Unicode whitespace handling (no-break space, em space, ideographic
  space).
* ``deduplicate`` count correctness on fully-unique and fully-duplicate
  inputs.
* ``canonicalization_fn_id`` and ``canonicalization_fn_version`` are
  exposed both as module constants and as attributes on the function
  itself.

These tests are example-based; the universal property (``Property 10:
Deduplication completeness``) is covered by a separate Hypothesis test in
task 4.7.

Why each test exists
--------------------

Each test cites the bullet from the task prompt or the requirement clause
it pins. A failure here surfaces a regression in either the documented
canonicalization rule or in the behavior the dataset card promises
downstream consumers.
"""

from __future__ import annotations

import pytest

from math_lora.dataset_builder import (
    CANONICALIZATION_FN_ID,
    CANONICALIZATION_FN_VERSION,
    TRAILING_PUNCTUATION,
    canonicalize,
    deduplicate,
)


# ===========================================================================
# canonicalize: individual rules
# ===========================================================================


class TestCanonicalizeLowercase:
    """The first step of the documented pipeline is :py:meth:`str.lower`."""

    @pytest.mark.unit
    def test_uppercase_input_is_lowercased(self) -> None:
        assert canonicalize("FIND X") == "find x"

    @pytest.mark.unit
    def test_mixed_case_input_is_lowercased(self) -> None:
        assert canonicalize("Find The Value Of X") == "find the value of x"

    @pytest.mark.unit
    def test_already_lowercase_input_is_unchanged_after_lowercasing(self) -> None:
        # No casing changes; only the documented post-lowercase steps may apply.
        assert canonicalize("solve for y") == "solve for y"

    @pytest.mark.unit
    def test_unicode_letters_are_lowercased(self) -> None:
        # ``str.lower`` is Unicode-aware; ÅB -> åb. This is documented as
        # part of the canonicalization rule.
        assert canonicalize("ÅB") == "åb"


class TestCanonicalizeTrim:
    """Trim leading and trailing whitespace per Requirement 3.4 step 3."""

    @pytest.mark.unit
    def test_leading_spaces_are_trimmed(self) -> None:
        assert canonicalize("   solve") == "solve"

    @pytest.mark.unit
    def test_trailing_spaces_are_trimmed(self) -> None:
        assert canonicalize("solve   ") == "solve"

    @pytest.mark.unit
    def test_leading_and_trailing_whitespace_are_trimmed_together(self) -> None:
        assert canonicalize("  \tsolve\n  ") == "solve"

    @pytest.mark.unit
    def test_input_that_is_only_whitespace_collapses_to_empty_string(self) -> None:
        # All whitespace -> after collapse becomes a single space, then
        # strip yields the empty string. The documented behavior.
        assert canonicalize("   \t\n  ") == ""


class TestCanonicalizeWhitespaceCollapse:
    """Internal whitespace runs are collapsed to a single ASCII space."""

    @pytest.mark.unit
    def test_double_spaces_collapse_to_single_space(self) -> None:
        assert canonicalize("find  x") == "find x"

    @pytest.mark.unit
    def test_long_run_of_spaces_collapses_to_single_space(self) -> None:
        assert canonicalize("a" + " " * 20 + "b") == "a b"

    @pytest.mark.unit
    def test_tab_run_collapses_to_single_space(self) -> None:
        assert canonicalize("a\t\tb") == "a b"

    @pytest.mark.unit
    def test_newline_run_collapses_to_single_space(self) -> None:
        # Newlines inside a problem are common when an ingest source
        # joined a multi-line problem statement; they are not significant.
        assert canonicalize("a\n\nb") == "a b"

    @pytest.mark.unit
    def test_mixed_whitespace_run_collapses_to_single_space(self) -> None:
        assert canonicalize("a \t\n\r\v\fb") == "a b"

    @pytest.mark.unit
    def test_collapse_runs_in_combination_with_trim(self) -> None:
        assert canonicalize("  \ta   \tb  ") == "a b"


class TestCanonicalizeUnicodeWhitespace:
    """Unicode whitespace is collapsed by the default ``re.UNICODE`` engine."""

    @pytest.mark.unit
    def test_no_break_space_is_treated_as_whitespace(self) -> None:
        # U+00A0 NO-BREAK SPACE is whitespace under Python ``\s``.
        assert canonicalize("a\u00a0b") == "a b"

    @pytest.mark.unit
    def test_em_space_is_treated_as_whitespace(self) -> None:
        # U+2003 EM SPACE.
        assert canonicalize("a\u2003b") == "a b"

    @pytest.mark.unit
    def test_ideographic_space_is_treated_as_whitespace(self) -> None:
        # U+3000 IDEOGRAPHIC SPACE -- common in Chinese-language sources.
        assert canonicalize("a\u3000b") == "a b"

    @pytest.mark.unit
    def test_mixed_unicode_and_ascii_whitespace_collapses(self) -> None:
        # No-break space + tab + em space all collapse to a single ASCII space.
        assert canonicalize("a\u00a0\t\u2003b") == "a b"

    @pytest.mark.unit
    def test_leading_unicode_whitespace_is_trimmed(self) -> None:
        assert canonicalize("\u00a0\u2003solve") == "solve"


class TestCanonicalizeTrailingPunctuation:
    """Each documented trailing-punctuation character is stripped."""

    @pytest.mark.unit
    def test_documented_set_matches_design(self) -> None:
        # Pin the exact set so a future broadening of the set is caught
        # by tests rather than slipping into a release silently.
        assert TRAILING_PUNCTUATION == frozenset({".", ",", "?", "!", ":", ";"})

    @pytest.mark.unit
    @pytest.mark.parametrize("punct", sorted(TRAILING_PUNCTUATION))
    def test_each_documented_trailing_punctuation_is_stripped(self, punct: str) -> None:
        assert canonicalize(f"hello{punct}") == "hello"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "tail",
        [
            "?!",
            "!?",
            ".,",
            ",.",
            ":;",
            ";:",
            "?!.",
            ".!?",
            "?,!:.",
        ],
    )
    def test_combinations_of_trailing_punctuation_are_stripped(self, tail: str) -> None:
        assert canonicalize(f"hello{tail}") == "hello"

    @pytest.mark.unit
    def test_internal_punctuation_is_preserved(self) -> None:
        # ``Solve:`` ends in a colon, but the colon is *internal* once a
        # following equation is appended; only trailing punctuation is
        # stripped, never internal punctuation.
        assert canonicalize("solve: 2x + 3 = 7") == "solve: 2x + 3 = 7"

    @pytest.mark.unit
    def test_trailing_punctuation_with_intermediate_whitespace_is_stripped(self) -> None:
        # After collapsing whitespace and trimming, the canonical form
        # ends in ``"hello?"`` then strip the ``?``.
        assert canonicalize("hello ?") == "hello"

    @pytest.mark.unit
    def test_trailing_punctuation_then_whitespace_is_stripped(self) -> None:
        # The whitespace is collapsed and trimmed first, then the
        # punctuation strip runs on the trimmed string.
        assert canonicalize("hello! ") == "hello"

    @pytest.mark.unit
    def test_punctuation_inside_a_word_is_not_stripped(self) -> None:
        # ``e.g.`` is a single token; only the final period would be
        # eligible for stripping at the end of the string.
        assert canonicalize("see e.g. theorem 1") == "see e.g. theorem 1"

    @pytest.mark.unit
    def test_non_documented_trailing_punctuation_is_preserved(self) -> None:
        # ``)`` and ``]`` are NOT in the documented set; they carry
        # mathematical meaning (closing a parenthesised expression) and
        # must be preserved.
        assert canonicalize("compute f(x)") == "compute f(x)"
        assert canonicalize("compute [a, b]") == "compute [a, b]"

    @pytest.mark.unit
    def test_math_operators_at_end_are_preserved(self) -> None:
        # ``=``, ``+``, ``-``, ``*``, ``/`` are not in the trailing set.
        for op in ("=", "+", "-", "*", "/"):
            assert canonicalize(f"x {op}") == f"x {op}", op


class TestCanonicalizeIdempotence:
    """``canonicalize(canonicalize(x)) == canonicalize(x)``."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "hello",
            "  Find  X.  ",
            "Solve: 2x + 3 = 7?!",
            "a\u00a0\t\u2003b",
            "what is x?",
            "WHAT IS Y!",
            "   \t\n  ",
            "compute f(x)",
            "see e.g. theorem 1",
        ],
    )
    def test_idempotent_on_diverse_inputs(self, raw: str) -> None:
        once = canonicalize(raw)
        twice = canonicalize(once)
        assert twice == once

    @pytest.mark.unit
    def test_canonical_form_is_stable_under_three_iterations(self) -> None:
        # Stronger than two-step idempotence: covers any pathological
        # interaction between trim and trailing-punctuation strip.
        once = canonicalize("  Find X?!  ")
        assert canonicalize(canonicalize(once)) == once


class TestCanonicalizeContracts:
    """Type contracts and edge cases."""

    @pytest.mark.unit
    def test_empty_string_returns_empty_string(self) -> None:
        assert canonicalize("") == ""

    @pytest.mark.unit
    def test_non_string_input_raises_type_error(self) -> None:
        # Documented in the function: silent coercion would create
        # false-positive dedup matches across malformed records.
        with pytest.raises(TypeError):
            canonicalize(None)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_non_string_int_input_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            canonicalize(123)  # type: ignore[arg-type]


# ===========================================================================
# Exposed canonicalization_fn_id / canonicalization_fn_version
# ===========================================================================


class TestCanonicalizationFnExposure:
    """The id and version are exposed for ``DatasetCard`` emission."""

    @pytest.mark.unit
    def test_module_level_id_constant_is_a_non_empty_string(self) -> None:
        assert isinstance(CANONICALIZATION_FN_ID, str)
        assert CANONICALIZATION_FN_ID  # non-empty

    @pytest.mark.unit
    def test_module_level_version_constant_is_a_non_empty_string(self) -> None:
        assert isinstance(CANONICALIZATION_FN_VERSION, str)
        assert CANONICALIZATION_FN_VERSION  # non-empty

    @pytest.mark.unit
    def test_function_attribute_id_matches_module_constant(self) -> None:
        # Required so callers that pass ``canonicalization_fn`` around
        # (e.g. the design's ``Dataset_Builder.build`` signature) can
        # read the id straight off the function reference.
        assert canonicalize.canonicalization_fn_id == CANONICALIZATION_FN_ID

    @pytest.mark.unit
    def test_function_attribute_version_matches_module_constant(self) -> None:
        assert canonicalize.canonicalization_fn_version == CANONICALIZATION_FN_VERSION

    @pytest.mark.unit
    def test_default_v1_id_is_the_documented_initial_value(self) -> None:
        # Lock the initial id so a rename of the published variant is
        # caught by tests; future variants must be additive.
        assert CANONICALIZATION_FN_ID == "default_v1"

    @pytest.mark.unit
    def test_initial_version_is_one_dot_zero(self) -> None:
        # Locks the initial release tag.
        assert CANONICALIZATION_FN_VERSION == "1.0"


# ===========================================================================
# deduplicate
# ===========================================================================


class TestDeduplicateBasics:
    """Behavior on small, hand-built inputs."""

    @pytest.mark.unit
    def test_empty_input_yields_empty_output_and_zero_dedup_count(self) -> None:
        unique, count = deduplicate([])
        assert unique == []
        assert count == 0

    @pytest.mark.unit
    def test_single_record_is_returned_unchanged_with_zero_dedup_count(self) -> None:
        records = [{"problem": "find x", "final_answer": "1"}]
        unique, count = deduplicate(records)
        assert unique == records
        assert count == 0


class TestDeduplicateFullyUnique:
    """All records are pairwise non-duplicates under canonicalization."""

    @pytest.mark.unit
    def test_fully_unique_input_yields_zero_dedup_count(self) -> None:
        records = [
            {"problem": f"problem number {i}", "final_answer": str(i)}
            for i in range(5)
        ]
        unique, count = deduplicate(records)
        assert count == 0
        assert len(unique) == len(records)

    @pytest.mark.unit
    def test_fully_unique_input_preserves_order(self) -> None:
        # First-seen-wins semantics imply the unique list is the input
        # list itself when no duplicates exist.
        records = [
            {"problem": "alpha"},
            {"problem": "beta"},
            {"problem": "gamma"},
        ]
        unique, _ = deduplicate(records)
        assert unique == records

    @pytest.mark.unit
    def test_fully_unique_problems_that_only_differ_by_internal_text_are_distinct(
        self,
    ) -> None:
        # ``find x`` vs ``find y`` -- a single internal character apart;
        # canonicalization doesn't merge these.
        records = [
            {"problem": "find x"},
            {"problem": "find y"},
        ]
        unique, count = deduplicate(records)
        assert count == 0
        assert len(unique) == 2


class TestDeduplicateFullyDuplicate:
    """Every record has the same canonical key."""

    @pytest.mark.unit
    def test_three_identical_records_dedup_to_one_with_count_two(self) -> None:
        records = [{"problem": "find x"}] * 3
        unique, count = deduplicate(records)
        assert len(unique) == 1
        assert count == 2

    @pytest.mark.unit
    def test_records_differing_only_by_case_dedup_to_one(self) -> None:
        # The canonical key is lowercase so case differences are
        # absorbed.
        records = [
            {"problem": "Find X"},
            {"problem": "FIND X"},
            {"problem": "find x"},
        ]
        unique, count = deduplicate(records)
        assert len(unique) == 1
        assert count == 2

    @pytest.mark.unit
    def test_records_differing_only_by_trailing_punctuation_dedup_to_one(self) -> None:
        records = [
            {"problem": "find x"},
            {"problem": "find x?"},
            {"problem": "find x!"},
            {"problem": "find x?!"},
        ]
        unique, count = deduplicate(records)
        assert len(unique) == 1
        assert count == 3

    @pytest.mark.unit
    def test_records_differing_only_by_whitespace_dedup_to_one(self) -> None:
        records = [
            {"problem": "find x"},
            {"problem": "  find  x"},
            {"problem": "find\tx"},
            {"problem": "find\nx"},
            {"problem": "find\u00a0x"},  # no-break space
        ]
        unique, count = deduplicate(records)
        assert len(unique) == 1
        assert count == 4

    @pytest.mark.unit
    def test_first_occurrence_is_kept(self) -> None:
        # Stability matters: the validation-split anti-leakage check
        # (Requirement 3.5) and Property 11 both depend on first-seen
        # being deterministic.
        first = {"problem": "Find X.", "final_answer": "first"}
        second = {"problem": "find x", "final_answer": "second"}
        third = {"problem": "FIND X!", "final_answer": "third"}
        unique, count = deduplicate([first, second, third])
        assert unique == [first]
        assert count == 2


class TestDeduplicateMixed:
    """Mixed inputs combining unique and duplicate keys."""

    @pytest.mark.unit
    def test_mixed_input_count_equals_input_minus_unique(self) -> None:
        # 5 inputs, 3 unique canonical keys -> count == 5 - 3 == 2.
        records = [
            {"problem": "find x"},
            {"problem": "FIND X"},   # dup of #1
            {"problem": "find y"},
            {"problem": "find y."},  # dup of #3
            {"problem": "find z"},
        ]
        unique, count = deduplicate(records)
        assert len(unique) == 3
        assert count == len(records) - len(unique)
        assert count == 2

    @pytest.mark.unit
    def test_mixed_input_preserves_first_occurrence_order(self) -> None:
        records = [
            {"problem": "alpha"},
            {"problem": "beta"},
            {"problem": "ALPHA"},  # dup
            {"problem": "gamma"},
            {"problem": "BETA!"},  # dup
        ]
        unique, _ = deduplicate(records)
        assert [r["problem"] for r in unique] == ["alpha", "beta", "gamma"]


class TestDeduplicateContracts:
    """Type and field contracts."""

    @pytest.mark.unit
    def test_record_without_problem_field_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            deduplicate([{"final_answer": "1"}])  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_record_with_non_string_problem_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            deduplicate([{"problem": 123, "final_answer": "1"}])  # type: ignore[list-item]

    @pytest.mark.unit
    def test_custom_canonicalization_fn_is_honored(self) -> None:
        # Stable test that injection works -- task 4.12 will use this
        # path to plug in the same canonicalize function explicitly.
        # The stub maps everything to the same key, so all records
        # collapse to the first.
        def collapse_all(_text: str) -> str:
            return "k"

        records = [{"problem": "alpha"}, {"problem": "beta"}, {"problem": "gamma"}]
        unique, count = deduplicate(records, canonicalization_fn=collapse_all)
        assert len(unique) == 1
        assert count == 2

    @pytest.mark.unit
    def test_records_iterable_is_consumed_only_once(self) -> None:
        # The implementation must not re-iterate ``records`` (real
        # callers pass generators).
        consumed: list[str] = []

        def gen():
            for s in ("a", "b", "a"):
                consumed.append(s)
                yield {"problem": s}

        unique, count = deduplicate(gen())
        assert len(unique) == 2
        assert count == 1
        # Generator was exhausted exactly once.
        assert consumed == ["a", "b", "a"]
