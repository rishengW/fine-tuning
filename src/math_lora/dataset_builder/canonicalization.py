"""Canonicalization and deduplication for ``Dataset_Builder`` (Requirement 3.4).

This module implements task **4.6** of the implementation plan: a stable,
documented ``canonicalize(text)`` function plus a ``deduplicate(records)``
helper that uses it to remove repeated training records by their canonical
``problem`` text. Both pieces are deliberately *pure* and free of upstream
dependencies (no ``transformers``, no ``datasets``) so they can be reused by
task **4.12** (``Custom_Integral_Set`` isolation) without circular imports
and so unit tests stay fast.

What canonicalization does
--------------------------

The default canonicalization function applies four transformations, in
order, all per Requirement 3.4 (and the design's *Dataset_Builder* §3.4
note). The order is fixed because each step assumes the previous one ran:

1. **Lowercase.** ``str.lower()`` Unicode-aware. Removes case as a source
   of duplicate-in-spirit problems (``"Find x"`` vs ``"find x"``).
2. **Whitespace collapse.** Every run of one or more whitespace characters
   -- as defined by the Python regex ``\\s`` flag (which matches ASCII
   spaces, tabs, newlines, and Unicode whitespace such as no-break space
   ``U+00A0``, em space ``U+2003``, ideographic space ``U+3000``, etc.) --
   is replaced by a single ASCII space ``" "``. This step is run *before*
   trimming so that "    \\t hello \\n  " collapses to " hello " and is
   then trimmed by the next step.
3. **Trim.** ``str.strip()`` removes any leading or trailing whitespace
   that survived the collapse step (mainly the single space that may
   remain at either end after collapse).
4. **Strip trailing punctuation.** A small, **documented** set of
   trailing punctuation characters is repeatedly removed from the right
   end of the string, together with any whitespace that gets exposed in
   between. The exact set is :data:`TRAILING_PUNCTUATION`. After this
   step, the string is trimmed once more to strip any whitespace exposed
   by punctuation removal (e.g. ``"hello ?"`` -> ``"hello"``).

The trailing-punctuation set
----------------------------

The set is small on purpose: it covers the punctuation commonly found at
the end of an English problem statement that does not change the
mathematical meaning. Math characters (``=``, ``<``, ``>``, ``+``, ``-``,
``*``, ``/``, ``)``, ``]``, ``}``, ``%``, etc.) and quote characters are
**not** in the set, because removing them would change the problem.

The set is:

.. data:: TRAILING_PUNCTUATION
   :noindex:

   ``frozenset({".", ",", "?", "!", ":", ";"})``

Stripping is repeated, so a problem ending in ``"?!"`` is canonicalized
the same as one ending in ``"!?"``.

Idempotence
-----------

The function is **idempotent**: ``canonicalize(canonicalize(x)) ==
canonicalize(x)`` for every input ``x``. This is what makes it safe to
call ``canonicalize`` again inside ``deduplicate`` without worrying about
input provenance, and what makes it safe for task 4.12 to call it on
``Custom_Integral_Set`` problems that may already have been canonicalized
during ingestion.

Versioning and dataset card integration
---------------------------------------

The function exports two module-level constants that are written verbatim
into every :class:`DatasetCard` per Requirement 3.7:

* :data:`CANONICALIZATION_FN_ID` -- a stable string identifier
  (``"default_v1"``) that names *which* canonicalization function was
  used. Future canonicalization variants will use a different id
  (``"default_v2"``, ``"strict_v1"``, ...) so an older dataset card can
  always be traced back to its exact canonicalization.
* :data:`CANONICALIZATION_FN_VERSION` -- a semver-style string
  (``"1.0"``) that bumps when the *behavior* of the function with that
  id changes. Within a single (id, version) pair the function is
  guaranteed to be deterministic and stable across releases.

The same pair is also exposed as attributes on :func:`canonicalize`
itself (``canonicalize.canonicalization_fn_id`` and
``canonicalize.canonicalization_fn_version``) so that downstream code
that already has the function reference -- e.g. task 4.14 emitting the
dataset card -- can pull both values directly off the function rather
than re-importing the constants.

If the trailing-punctuation set, the whitespace-collapse rule, or the
ordering ever changes, the version string in this module **MUST** bump,
and a new id may be needed if the change is not backward-compatible. See
the *Determinism* note in the module docstring of
:mod:`math_lora.dataset_builder.sources` for the matching rationale on
ingestion connectors.

Reused by task 4.12
-------------------

Task 4.12 ("Custom_Integral_Set isolation") and task 4.8 ("validation
split anti-leakage") both compare problems by their canonical form. They
import :func:`canonicalize` directly from this module to guarantee that
the comparison key is bit-identical across all three usages (dedup,
isolation, val-split).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Final, TypeVar


# ---------------------------------------------------------------------------
# Documented constants
# ---------------------------------------------------------------------------

#: Stable identifier for *this* canonicalization variant. Recorded verbatim
#: in :class:`DatasetCard.canonicalization_fn_id` (Requirement 3.7) so that
#: a dataset card can be replayed on the exact same canonicalization.
#:
#: The id is **opaque** -- it is not parsed by any consumer; do not embed
#: behavior in the string. To change behavior, mint a new id (or bump the
#: version, depending on whether the change is backward-compatible) and
#: leave existing dataset cards pointing at the old id untouched.
CANONICALIZATION_FN_ID: Final[str] = "default_v1"

#: Semantic-version-style behavior tag for the function published under
#: :data:`CANONICALIZATION_FN_ID`. Bumped whenever the observable behavior
#: of the function changes (e.g. trailing-punctuation set is extended, or
#: a new normalization step is added). Recorded in
#: :class:`DatasetCard.canonicalization_fn_version`.
CANONICALIZATION_FN_VERSION: Final[str] = "1.0"

#: The exact set of trailing punctuation characters stripped by
#: :func:`canonicalize`. Documented per Requirement 3.4 ("a documented set
#: of trailing punctuation"). The set is:
#:
#: * ``"."`` -- sentence-ending period.
#: * ``","`` -- comma (sometimes appears before a list of givens).
#: * ``"?"`` -- question mark (most word problems end in one).
#: * ``"!"`` -- exclamation mark.
#: * ``":"`` -- colon (e.g. ``"Find x:"`` before a list of equations).
#: * ``";"`` -- semicolon.
#:
#: All characters are ASCII; Unicode variants (e.g. ``U+FF1F`` fullwidth
#: question mark) are **not** in the set because they appear so rarely in
#: the target corpora that adding them would create false negatives more
#: than true matches. If a future corpus needs them, mint a new
#: ``CANONICALIZATION_FN_ID``.
TRAILING_PUNCTUATION: Final[frozenset[str]] = frozenset({".", ",", "?", "!", ":", ";"})


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

# A single regex that matches one or more whitespace characters. ``\s`` in
# Python's ``re`` module is Unicode-aware by default on ``str`` inputs --
# it matches ``[ \t\n\r\f\v]`` plus every character with Unicode category
# ``Zs`` (no-break space, em space, ideographic space, etc.) when the
# default ``re.UNICODE`` flag is in effect, which it is for ``str``
# patterns since Python 3. We compile it once so :func:`canonicalize` can
# stay allocation-free for short inputs.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")


def _strip_trailing_punctuation(text: str) -> str:
    """Strip every trailing character in :data:`TRAILING_PUNCTUATION`.

    Whitespace exposed *between* punctuation characters during stripping
    is also removed, so ``"hello! ?"`` becomes ``"hello"``: first the
    trailing ``"?"`` is stripped, then the now-trailing whitespace, then
    the now-trailing ``"!"``. The loop terminates when neither a
    whitespace nor a punctuation character is at the end.

    The implementation is O(len(text)) in the worst case (a string that
    is entirely whitespace and trailing punctuation) and O(1) when the
    string ends in a non-punctuation character, which is the common case.
    """

    # Use ``rstrip`` rather than reverse-iteration because ``rstrip``
    # is implemented in C and is much faster than a per-character Python
    # loop. The two calls alternate until neither shortens the string.
    punct_chars = "".join(TRAILING_PUNCTUATION)
    while True:
        before = text
        text = text.rstrip()  # whitespace exposed by previous punct strip
        text = text.rstrip(punct_chars)
        if text == before:
            return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canonicalize(text: str) -> str:
    """Return the canonical form of ``text`` per Requirement 3.4.

    Performs, in order:

    1. Lowercase via :py:meth:`str.lower`.
    2. Collapse every Unicode-whitespace run to a single ASCII space.
    3. Trim leading and trailing whitespace.
    4. Strip every trailing character in :data:`TRAILING_PUNCTUATION`,
       repeatedly, together with whitespace that becomes exposed between
       punctuation removals.

    The function is **deterministic**, **pure**, and **idempotent**:
    ``canonicalize(canonicalize(x)) == canonicalize(x)`` for every
    string ``x``. It is the single comparison key used by

    * :func:`deduplicate` (Requirement 3.4),
    * the validation-split anti-leakage check (Requirement 3.5),
    * the ``Custom_Integral_Set`` exclusion (Requirement 3.10),

    so changing its behavior changes the deduplication semantics of the
    whole pipeline. If the behavior must change, bump
    :data:`CANONICALIZATION_FN_VERSION` (or mint a new
    :data:`CANONICALIZATION_FN_ID` for breaking changes) so that older
    dataset cards remain replayable.

    Args:
        text: Any string. ``""`` is allowed and yields ``""``. Surrogate
            and non-BMP characters pass through unchanged after
            lowercasing (Python's ``str.lower`` handles them correctly).

    Returns:
        The canonical form of ``text`` as described above.

    Raises:
        TypeError: If ``text`` is not a :class:`str`. We surface this
            eagerly rather than coercing because the deduplication key
            silently swallowing non-string inputs (e.g. ``None`` from a
            malformed source row) would create false-positive matches
            across unrelated records.

    Examples:
        >>> canonicalize("  Find  X.  ")
        'find x'
        >>> canonicalize("Solve: 2x + 3 = 7?!")
        'solve: 2x + 3 = 7'
        Note the leading ``Solve:`` -- the colon is *internal*, not
        trailing, so it stays. Only the ``"?!"`` at the end is stripped.

        >>> canonicalize("hello") == canonicalize(canonicalize("hello"))
        True
    """

    if not isinstance(text, str):
        raise TypeError(
            f"canonicalize: expected str, got {type(text).__name__}"
        )

    # 1. Lowercase. Done first so that any later string comparisons
    #    against ``TRAILING_PUNCTUATION`` (all ASCII) are unaffected by
    #    case. ``str.lower`` is Unicode-aware.
    text = text.lower()

    # 2. Collapse whitespace runs. The regex matches ``\s+`` against the
    #    Unicode-default ``re`` engine so both ASCII whitespace and
    #    characters in Unicode category Zs (no-break space, em space,
    #    ideographic space, ...) are collapsed.
    text = _WHITESPACE_RUN.sub(" ", text)

    # 3. Trim. After step 2 the only whitespace left is single ASCII
    #    spaces; ``str.strip`` removes them at both ends.
    text = text.strip()

    # 4. Strip trailing punctuation (and any whitespace exposed during
    #    stripping). See :func:`_strip_trailing_punctuation` for details.
    text = _strip_trailing_punctuation(text)

    return text


# Expose the id/version on the function object itself so callers that
# already have a reference to the function can read both values without
# re-importing the module-level constants. Useful for task 4.14
# (DatasetCard emission), which receives a ``canonicalization_fn``
# parameter per the design's ``Dataset_Builder.build`` signature.
canonicalize.canonicalization_fn_id = CANONICALIZATION_FN_ID  # type: ignore[attr-defined]
canonicalize.canonicalization_fn_version = CANONICALIZATION_FN_VERSION  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


# A record is anything mapping-like that has a ``problem`` key whose value
# is a string. ``Mapping[str, object]`` is loose on purpose: real records
# carry many fields (``solution_steps``, ``final_answer``, ``raw``) and
# we do not want this helper to dictate which extra fields are allowed.
RecordT = TypeVar("RecordT", bound=Mapping[str, object])


def deduplicate(
    records: Iterable[RecordT],
    *,
    canonicalization_fn: Callable[[str], str] = canonicalize,
) -> tuple[list[RecordT], int]:
    """Deduplicate ``records`` by canonicalized ``problem`` text.

    Implements Requirement 3.4: deduplicate the ingested-and-normalized
    records by applying ``canonicalization_fn`` to each record's
    ``problem`` field and keeping the **first** record observed for each
    canonical key.

    Stability:
        First-seen wins. Iterating ``records`` again over the same input
        yields the same output (provided ``records`` is itself
        deterministic). This matches Property 11's requirement that the
        validation split be byte-identical across runs with the same
        seed -- the dedup step must not reorder records.

    Args:
        records: Any iterable of mappings. Each mapping MUST contain a
            ``"problem"`` key whose value is a string. Records lacking
            ``"problem"`` raise ``KeyError``; records with a non-string
            ``"problem"`` raise ``TypeError`` (via
            :func:`canonicalize`). This is intentional: the
            normalization layer (task 4.3) is supposed to have
            already-validated records before they reach dedup, so
            anything malformed at this point is a programming error,
            not user data.
        canonicalization_fn: The function used to derive the dedup key
            from each record's ``problem`` text. Defaults to
            :func:`canonicalize`. The signature is left injectable so
            tests can substitute a stub without monkey-patching the
            module, and so a future ``canonicalization_fn`` variant
            (e.g. ``default_v2``) can be plugged in without changing
            this helper.

    Returns:
        A 2-tuple ``(unique_records, dedup_count)``:

        * ``unique_records`` -- a ``list`` of records with the same item
          type as the input, preserving the order of first occurrence.
        * ``dedup_count`` -- ``len(input_records) - len(unique_records)``,
          the number of records dropped. This is the value written into
          :class:`DatasetCard.record_count_after_dedup` indirectly (the
          card stores the post-dedup count; ``dedup_count`` is the
          difference) and validated by Property 10 (Deduplication
          completeness).

    Examples:
        >>> recs = [
        ...     {"problem": "Find x.", "final_answer": "1"},
        ...     {"problem": "find x", "final_answer": "1"},  # duplicate
        ...     {"problem": "Find y.", "final_answer": "2"},
        ... ]
        >>> unique, count = deduplicate(recs)
        >>> [r["problem"] for r in unique]
        ['Find x.', 'Find y.']
        >>> count
        1
    """

    seen: set[str] = set()
    unique: list[RecordT] = []
    total = 0

    for record in records:
        total += 1
        # ``record["problem"]`` deliberately raises KeyError for missing
        # field -- see docstring rationale.
        problem = record["problem"]
        if not isinstance(problem, str):
            raise TypeError(
                "deduplicate: record['problem'] must be str, got "
                f"{type(problem).__name__}"
            )
        key = canonicalization_fn(problem)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    dedup_count = total - len(unique)
    return unique, dedup_count


__all__ = [
    "CANONICALIZATION_FN_ID",
    "CANONICALIZATION_FN_VERSION",
    "TRAILING_PUNCTUATION",
    "canonicalize",
    "deduplicate",
]
