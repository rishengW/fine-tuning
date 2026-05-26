"""Reasoning_Format normalization for ``Dataset_Builder`` (Requirement 3.2, 3.3, 3.6).

This module is task 4.3 in
``.kiro/specs/math-lora-finetuning/tasks.md``: it converts the best-effort
:class:`~math_lora.dataset_builder.sources.RawRecord` values yielded by the
connector layer (task 4.1) into the strict ``Reasoning_Format`` carried by
:class:`~math_lora.types.ReasoningRecord`.

What this module does
---------------------

* :func:`normalize_record` -- pure function that takes a single dict-like
  raw record and returns either a fully-validated ``ReasoningRecord`` or a
  structured :class:`RejectedRecord` describing *why* the record was
  refused (per Requirement 3.3).
* :func:`normalize_records` -- batch helper that drives
  :func:`normalize_record` over an iterable and returns
  ``(accepted_records, rejection_reason_counts)``. The returned counts dict
  is what the dataset accounting balance (task 4.5 / Property 9) hinges on:
  ``len(input) == len(accepted) + sum(rejection_reason_counts.values())``.

What this module deliberately does NOT do
-----------------------------------------

* No deduplication. That belongs to task 4.6, which canonicalizes
  ``problem`` text and counts dedup hits separately.
* No whitespace stripping or LaTeX munging on field values. Requirement
  3.6 requires that LaTeX delimiters (``$...$``, ``\\(...\\)``, ``\\[...\\]``,
  ``$$...$$``) and mathematical notation are preserved byte-for-byte; the
  emptiness checks below use ``.strip()`` only as a *predicate* on whether
  the string carries any non-whitespace content -- the original field
  bytes are passed through untouched into the resulting ``ReasoningRecord``.
* No tokenizer-aware truncation. That is task 4.10 and is concerned with
  ``max_seq_len`` rather than per-record validity.

Rejection reasons
-----------------

Three reasons cover Requirement 3.3 exactly:

* :data:`REASON_EMPTY_PROBLEM` -- ``problem`` is missing, empty, or
  whitespace-only.
* :data:`REASON_EMPTY_SOLUTION_STEPS` -- ``solution_steps`` is missing,
  zero-length, or contains no non-whitespace entries.
* :data:`REASON_EMPTY_FINAL_ANSWER` -- ``final_answer`` is missing, empty,
  or whitespace-only.

The reasons are exported as module-level string constants so that callers
in later tasks (the dataset-card emitter in task 4.14) can reference them
without depending on ``RejectedRecord`` literal values.

Error handling
--------------

Per the design's *Error Handling* table, schema-shaped problems surface as
exceptions, but a malformed *value* of a known field is **not** an error
here -- it is a normal rejection counted toward the reason map. That keeps
the dataset accounting balance trivially provable: every input either
becomes one accepted record or contributes exactly one count to the
rejection map. If the input is not a dict-like object at all (e.g. a
generator yields ``None`` because of a connector bug), :class:`TypeError`
is raised so the bug surfaces immediately rather than silently inflating
the rejection counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Union

from math_lora.types import ReasoningRecord


# ---------------------------------------------------------------------------
# Rejection reason constants
# ---------------------------------------------------------------------------


#: ``problem`` field is missing, empty, or whitespace-only (Requirement 3.3).
REASON_EMPTY_PROBLEM: Final[str] = "empty_problem"

#: ``solution_steps`` is missing/None, zero-length, or has no non-whitespace
#: entries (Requirement 3.3).
REASON_EMPTY_SOLUTION_STEPS: Final[str] = "empty_solution_steps"

#: ``final_answer`` is missing, empty, or whitespace-only (Requirement 3.3).
REASON_EMPTY_FINAL_ANSWER: Final[str] = "empty_final_answer"


#: Tuple of every rejection reason this module can emit. Used by tests and by
#: the dataset-card emitter (task 4.14) to seed a reasons map with zero counts
#: so that downstream code can rely on every key being present.
ALL_REJECTION_REASONS: Final[tuple[str, ...]] = (
    REASON_EMPTY_PROBLEM,
    REASON_EMPTY_SOLUTION_STEPS,
    REASON_EMPTY_FINAL_ANSWER,
)


# ---------------------------------------------------------------------------
# Rejection result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectedRecord:
    """Structured rejection produced by :func:`normalize_record`.

    Attributes:
        reason: One of :data:`ALL_REJECTION_REASONS`. The string form is
            stable across releases so dataset-card consumers can key on it.
        raw: The original input dict, retained so that downstream tooling
            (dataset-card emission, debugging) can reference the offending
            row without having to re-scan the input. The dict is shallow-
            copied at construction time to defend against caller mutation.
    """

    reason: str
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Defensive shallow copy: the caller may continue mutating the
        # dict it passed in, but the rejection record should be immutable
        # from the caller's point of view to keep the rejection map stable.
        # ``object.__setattr__`` is required because the dataclass is frozen.
        if not isinstance(self.raw, Mapping):
            raise TypeError(
                f"RejectedRecord.raw must be a Mapping, got {type(self.raw).__name__}"
            )
        object.__setattr__(self, "raw", dict(self.raw))


#: Result type returned by :func:`normalize_record`.
NormalizationResult = Union[ReasoningRecord, RejectedRecord]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_blank(value: object) -> bool:
    """Return ``True`` when ``value`` carries no non-whitespace content.

    Treats non-string values as blank so a connector that accidentally
    yields ``None`` or ``42`` for a string field falls through to a
    rejection rather than crashing the batch. Strings are checked via
    ``.strip()`` so whitespace-only entries (``"  "``, ``"\\n\\t"``) count
    as blank per the Requirement 3.2 / 3.3 reading of "non-empty".
    """

    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _filter_steps(steps_value: object) -> list[str] | None:
    """Reduce ``steps_value`` to a list of non-blank string steps.

    Returns ``None`` when the input is missing or zero-length after
    filtering. The original byte content of each surviving step is
    preserved verbatim (no ``.strip()``, no LaTeX rewrite) so Requirement
    3.6 holds.

    Accepts:

    * ``None`` -> ``None``
    * ``list[str]`` -> filtered list, dropping whitespace-only entries;
      ``None`` if everything was filtered out.
    * Any other type -> ``None`` (treated as missing, surfaces as
      :data:`REASON_EMPTY_SOLUTION_STEPS`).

    Non-string elements inside a list are silently ignored: a connector
    bug that yields ``[None, "valid step"]`` will accept the valid step
    and drop the ``None``. The alternative (raising) would let a single
    malformed element poison the whole batch, which contradicts the
    "record reason counts" framing of Requirement 3.3.
    """

    if steps_value is None:
        return None
    if not isinstance(steps_value, list):
        return None
    surviving: list[str] = []
    for entry in steps_value:
        if isinstance(entry, str) and entry.strip() != "":
            surviving.append(entry)
    return surviving if surviving else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_record(raw: Mapping[str, Any]) -> NormalizationResult:
    """Normalize a single raw record into a ``ReasoningRecord`` or rejection.

    Pure: deterministic on its input and free of I/O / global state.

    The raw record is expected to follow the
    :class:`~math_lora.dataset_builder.sources.RawRecord` shape but the
    function is intentionally tolerant -- any ``Mapping`` with the right
    keys is accepted, and any missing key surfaces as the appropriate
    rejection reason rather than as an exception. This keeps the public
    contract usable for operator-supplied records that arrive directly
    from a JSON loader or a dict literal in a unit test.

    Args:
        raw: A mapping with the keys ``"problem"``, ``"solution_steps"``,
            and ``"final_answer"``. Extra keys are ignored. Missing keys
            are treated as if their value were absent / blank and rejected
            with the appropriate reason.

    Returns:
        Either a :class:`~math_lora.types.ReasoningRecord` -- when every
        field is non-empty in the spec sense -- or a :class:`RejectedRecord`
        whose ``reason`` is one of :data:`ALL_REJECTION_REASONS`.

    Raises:
        TypeError: If ``raw`` is not a ``Mapping``. Connector bugs that
            yield non-mapping values (``None``, lists, primitives) should
            surface immediately rather than inflate the rejection counts.

    Notes:
        Rejection precedence is fixed and documented to be deterministic:
        ``problem`` is checked first, then ``solution_steps``, then
        ``final_answer``. The order is stable so that dataset cards
        produced from two runs over the same input record yield identical
        reason counts (relevant to the reproducibility guarantees in
        Requirement 8.10).

        LaTeX preservation (Requirement 3.6) is enforced by passing the
        original field values into ``ReasoningRecord`` without any
        whitespace stripping, normalization, or transformation. The
        ``_is_blank`` predicate uses ``.strip()`` only to *test* for
        emptiness; the original bytes are copied through unchanged.
    """

    if not isinstance(raw, Mapping):
        raise TypeError(
            f"normalize_record: raw must be a Mapping, got {type(raw).__name__}"
        )

    # Field 1: problem -- reject before looking at the others so the reason
    # for rejection is the *first* offending field. This deterministic
    # precedence means two runs over the same input produce the same
    # reason counts (matters for Requirement 8.10).
    problem_value = raw.get("problem")
    if _is_blank(problem_value):
        return RejectedRecord(reason=REASON_EMPTY_PROBLEM, raw=raw)

    # Field 2: solution_steps. Preserve the surviving steps' original bytes;
    # whitespace-only entries are filtered out so a single ``"  "`` step
    # interleaved among real steps does not poison the whole record. If
    # the filter leaves nothing behind, reject the record under
    # ``empty_solution_steps``.
    steps_value = raw.get("solution_steps")
    filtered_steps = _filter_steps(steps_value)
    if filtered_steps is None:
        return RejectedRecord(reason=REASON_EMPTY_SOLUTION_STEPS, raw=raw)

    # Field 3: final_answer.
    final_answer_value = raw.get("final_answer")
    if _is_blank(final_answer_value):
        return RejectedRecord(reason=REASON_EMPTY_FINAL_ANSWER, raw=raw)

    # All three fields are non-empty in the spec sense; build the strict
    # ReasoningRecord. ``ReasoningRecord.parse`` enforces the same
    # invariants we just hand-checked, so a defect here would surface as
    # a SchemaValidationError rather than silently producing a malformed
    # record. We rely on the type narrowings above to avoid type: ignore.
    assert isinstance(problem_value, str)  # narrowed by _is_blank check
    assert isinstance(final_answer_value, str)  # narrowed by _is_blank check
    return ReasoningRecord(
        problem=problem_value,
        solution_steps=filtered_steps,
        final_answer=final_answer_value,
    )


def normalize_records(
    raws: Iterable[Mapping[str, Any]],
) -> tuple[list[ReasoningRecord], dict[str, int]]:
    """Normalize an iterable of raw records into accepted + reason counts.

    This is the helper that downstream dataset accounting (task 4.5,
    Property 9) builds on. The invariant ::

        len(input_records) == len(accepted_records) + sum(rejection_reasons.values())

    holds by construction for every input where every element is a
    ``Mapping``: each iteration step contributes exactly one outcome --
    either an accepted record or a single rejection-count increment.

    Args:
        raws: Any iterable of mapping-like raw records. The function does
            not retain a reference to the iterable after it returns; if
            you need to consume it twice, pass a list.

    Returns:
        A two-tuple ``(accepted_records, rejection_reason_counts)``:

        * ``accepted_records``: list of :class:`ReasoningRecord` instances
          in the same order as the input iterable (only accepted records
          appear). Order matters for the deterministic train/val split in
          task 4.8.
        * ``rejection_reason_counts``: dict keyed by every reason in
          :data:`ALL_REJECTION_REASONS`. Reasons with zero hits are still
          present in the dict (value ``0``), so dataset-card consumers
          can index by reason without ``.get(...)`` boilerplate.

    Raises:
        TypeError: If any element of ``raws`` is not a ``Mapping``. The
            error fires on the first offending element; preceding elements
            are still consumed, but the function does not return a
            partial result -- callers must fix the producer.
    """

    accepted: list[ReasoningRecord] = []
    # Pre-seed every reason at zero so consumers can index by reason name
    # without checking for membership. The total sum is unaffected.
    reason_counts: dict[str, int] = dict.fromkeys(ALL_REJECTION_REASONS, 0)

    for raw in raws:
        result = normalize_record(raw)
        if isinstance(result, ReasoningRecord):
            accepted.append(result)
        else:
            # Defensive sanity check: every reason emitted by
            # ``normalize_record`` MUST be in ``ALL_REJECTION_REASONS``,
            # otherwise the dataset accounting balance breaks. The check
            # is cheap and catches programmer errors immediately.
            if result.reason not in reason_counts:
                raise AssertionError(
                    f"normalize_records: unexpected rejection reason "
                    f"{result.reason!r}; expected one of "
                    f"{ALL_REJECTION_REASONS!r}"
                )
            reason_counts[result.reason] += 1

    return accepted, reason_counts


__all__: Final = [
    "REASON_EMPTY_PROBLEM",
    "REASON_EMPTY_SOLUTION_STEPS",
    "REASON_EMPTY_FINAL_ANSWER",
    "ALL_REJECTION_REASONS",
    "RejectedRecord",
    "NormalizationResult",
    "normalize_record",
    "normalize_records",
]
