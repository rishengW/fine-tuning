"""Smoke tests for the four ``Dataset_Builder`` source connectors (task 4.2).

Coverage matrix
---------------

Requirement 3.1 enumerates four ingest sources that the
``Dataset_Builder`` must support. This file ships exactly **one** smoke
test per source -- the minimum required by the task -- with each test
following a two-step recipe:

1. **Interface availability.** The connector class is importable from the
   public ``math_lora.dataset_builder`` namespace, conforms to the
   :class:`~math_lora.dataset_builder.sources.DatasetSource` protocol
   (``runtime_checkable``), and exposes the documented attributes
   (``source_id``, ``license``).
2. **Small fixture round-trip.** A tiny fixture (either a bundled JSONL
   file or an inline ``records=[...]`` list, per the connector's
   contract) is fed to :meth:`iter_records`, and the yielded
   :class:`~math_lora.dataset_builder.sources.RawRecord` values are
   asserted to round-trip the source fields the connector is responsible
   for extracting.

What these tests are *not*
~~~~~~~~~~~~~~~~~~~~~~~~~~

* They do not exercise normalization (Requirement 3.2), deduplication
  (Requirement 3.4), splitting (Requirement 3.5), tokenizer-aware
  truncation (Requirement 3.8), exclusion (Requirement 3.10), or
  dataset-card emission (Requirement 3.7). Those concerns belong to
  later tasks (4.3, 4.6, 4.8, 4.10, 4.12, 4.14) which carry their own
  property and unit tests.
* They do not hit the real ``datasets`` library or any remote endpoint.
  Per the connector module's docstring, network-backed connectors are
  driven by local fixtures in CI so the smoke tests stay deterministic
  and offline -- the deferred real-fetch path is documented to live in
  later integration tests.

Fixture sources
~~~~~~~~~~~~~~~

The bundled JSONL fixtures live at:

* ``tests/fixtures/datasets/gsm8k_sample.jsonl``
* ``tests/fixtures/datasets/math_sample.jsonl``

For the open step-by-step corpus and operator-supplied integral pairs
there is no upstream public fixture, so the tests build a 2-record
inline list that satisfies the explicit ``Reasoning_Format`` shape
(Requirement 3.2) those two connectors expect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from math_lora.dataset_builder import (
    DatasetSource,
    GSM8KTrainSource,
    MATHTrainSource,
    OpenStepByStepSource,
    OperatorIntegralSource,
)


# ---------------------------------------------------------------------------
# Fixture path resolution
# ---------------------------------------------------------------------------
#
# ``__file__`` is ``tests/integration/test_source_connectors_smoke.py``;
# the bundled JSONL fixtures live two levels up under
# ``tests/fixtures/datasets/``. Resolving from ``__file__`` keeps the
# tests robust to ``pytest`` being invoked from any working directory.
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "datasets"
_GSM8K_FIXTURE = _FIXTURES_DIR / "gsm8k_sample.jsonl"
_MATH_FIXTURE = _FIXTURES_DIR / "math_sample.jsonl"


@pytest.fixture(scope="module")
def gsm8k_fixture_path() -> Path:
    """Return the bundled GSM8K JSONL fixture path, asserting it exists."""

    assert _GSM8K_FIXTURE.is_file(), (
        f"GSM8K smoke fixture missing at {_GSM8K_FIXTURE}; the smoke test "
        f"requires the bundled fixture from task 1.1's repo layout."
    )
    return _GSM8K_FIXTURE


@pytest.fixture(scope="module")
def math_fixture_path() -> Path:
    """Return the bundled MATH JSONL fixture path, asserting it exists."""

    assert _MATH_FIXTURE.is_file(), (
        f"MATH smoke fixture missing at {_MATH_FIXTURE}; the smoke test "
        f"requires the bundled fixture from task 1.1's repo layout."
    )
    return _MATH_FIXTURE


# ===========================================================================
# Source 1: GSM8K training split (Requirement 3.1, bullet 1)
# ===========================================================================


@pytest.mark.integration
def test_gsm8k_train_source_smoke(gsm8k_fixture_path: Path) -> None:
    """Smoke test for :class:`GSM8KTrainSource` (Requirement 3.1).

    Exercises the connector against the bundled
    ``tests/fixtures/datasets/gsm8k_sample.jsonl`` fixture, which has 3
    rows in the published GSM8K shape ``{"question": ..., "answer":
    "<reasoning>\\n#### <final>"}``. Verifies that:

    * The class implements :class:`DatasetSource` at runtime.
    * ``source_id`` and ``license`` carry the documented defaults.
    * ``record_count()`` matches the fixture line count (round-trip).
    * Every yielded :class:`RawRecord` carries the verbatim ``question``
      as ``problem``, the post-``####`` text as ``final_answer``, a
      non-empty ``solution_steps`` list (the connector splits on ``\\n``
      and drops blank lines), and a ``raw`` field that mirrors the input
      dict.
    """

    source = GSM8KTrainSource(fixture_path=gsm8k_fixture_path)

    # Step 1: interface availability.
    assert isinstance(source, DatasetSource)
    assert source.source_id == "gsm8k_train"
    assert source.license == "gsm8k-mit"
    assert source.record_count() == 3

    # Step 2: fixture round-trip.
    records = list(source.iter_records())
    assert len(records) == 3

    # First fixture row: "Janet has 3 apples..." -> final answer "5".
    first = records[0]
    assert first["problem"].startswith("Janet has 3 apples")
    assert first["final_answer"] == "5"
    assert first["solution_steps"] is not None
    assert len(first["solution_steps"]) >= 1
    # The connector must preserve the source dict verbatim under ``raw``
    # so that DatasetCard.rejection_reasons (Requirement 3.7) can refer
    # back to the original row.
    assert first["raw"]["question"] == first["problem"]
    assert "####" in first["raw"]["answer"]

    # Every record gets a non-empty extracted final_answer because all
    # three fixture rows contain the ``####`` delimiter.
    for record in records:
        assert record["final_answer"], (
            "GSM8K connector failed to extract #### final answer "
            "from fixture row"
        )
        assert record["problem"], "GSM8K connector dropped problem text"


# ===========================================================================
# Source 2: MATH training split (Requirement 3.1, bullet 2)
# ===========================================================================


@pytest.mark.integration
def test_math_train_source_smoke(math_fixture_path: Path) -> None:
    """Smoke test for :class:`MATHTrainSource` (Requirement 3.1).

    Exercises the connector against the bundled
    ``tests/fixtures/datasets/math_sample.jsonl`` fixture, which has 3
    rows in the published MATH shape ``{"problem": ..., "solution":
    "... \\boxed{<final>} ..."}``. Verifies that:

    * The class implements :class:`DatasetSource` at runtime.
    * ``source_id`` and ``license`` carry the documented defaults.
    * ``record_count()`` matches the fixture line count.
    * ``\\boxed{...}`` extraction yields the expected final answers.
    * LaTeX delimiters survive round-trip per Requirement 3.6 -- the
      ``problem`` text retains its ``$...$`` math mode markers.
    """

    source = MATHTrainSource(fixture_path=math_fixture_path)

    # Step 1: interface availability.
    assert isinstance(source, DatasetSource)
    assert source.source_id == "math_train"
    assert source.license == "math-cc-by-sa-4.0"
    assert source.record_count() == 3

    # Step 2: fixture round-trip with \boxed{...} extraction.
    records = list(source.iter_records())
    assert len(records) == 3

    # The three fixture rows produce ``\boxed{4}``, ``\boxed{2x}``, and
    # ``\boxed{\frac{1}{2}}``; the connector regex extracts the inner
    # group with surrounding whitespace stripped. The third row tests
    # that LaTeX inside the box (``\frac{1}{2}``) is preserved so
    # downstream symbolic_equivalence scoring (Requirement 6.7) sees
    # the same expression the corpus authored.
    expected_final_answers = ["4", "2x", "\\frac{1}{2}"]
    actual_final_answers = [r["final_answer"] for r in records]
    assert actual_final_answers == expected_final_answers

    # Requirement 3.6: LaTeX delimiters survive ingest. The first row's
    # problem contains ``$2x + 3 = 11$``; the connector must not strip
    # the ``$...$`` math-mode markers.
    first = records[0]
    assert "$" in first["problem"], (
        "MATH connector stripped LaTeX delimiters from problem text "
        "(violates Requirement 3.6)"
    )
    assert first["solution_steps"] is not None
    assert len(first["solution_steps"]) >= 1

    # Every record's raw dict must round-trip the input fields.
    for record in records:
        assert record["raw"]["problem"] == record["problem"]
        assert "\\boxed{" in record["raw"]["solution"]


# ===========================================================================
# Source 3: open-source step-by-step corpus (Requirement 3.1, bullet 3)
# ===========================================================================


@pytest.mark.integration
def test_open_step_by_step_source_smoke() -> None:
    """Smoke test for :class:`OpenStepByStepSource` (Requirement 3.1).

    The open step-by-step corpus is expected to ship records already in
    the explicit ``Reasoning_Format`` shape (Requirement 3.2)::

        {"problem": str, "solution_steps": [str, ...], "final_answer": str}

    so the connector is a passthrough -- this test feeds 2 inline
    records, verifies the connector copies each field verbatim, and
    confirms the protocol surface. There is no public bundled fixture
    for this corpus (operators choose their own source); using inline
    records keeps the smoke test deterministic and operator-agnostic.
    """

    inline_records = [
        {
            "problem": "Solve for x: 2x = 10",
            "solution_steps": [
                "Divide both sides by 2.",
                "x = 5",
            ],
            "final_answer": "5",
        },
        {
            "problem": "Differentiate f(x) = sin(x).",
            "solution_steps": [
                "By the standard derivative table, d/dx sin(x) = cos(x).",
            ],
            "final_answer": "cos(x)",
        },
    ]
    source = OpenStepByStepSource(records=inline_records)

    # Step 1: interface availability.
    assert isinstance(source, DatasetSource)
    assert source.source_id == "open_step_by_step"
    assert source.license == "cc-by-4.0"
    assert source.record_count() == 2

    # Step 2: fixture round-trip. The connector must copy every explicit
    # ``Reasoning_Format`` field verbatim and leave validation to task 4.3.
    records = list(source.iter_records())
    assert len(records) == 2

    first = records[0]
    assert first["problem"] == "Solve for x: 2x = 10"
    assert first["solution_steps"] == [
        "Divide both sides by 2.",
        "x = 5",
    ]
    assert first["final_answer"] == "5"

    second = records[1]
    assert second["problem"] == "Differentiate f(x) = sin(x)."
    assert second["solution_steps"] == [
        "By the standard derivative table, d/dx sin(x) = cos(x).",
    ]
    assert second["final_answer"] == "cos(x)"

    # Iterating twice must yield the same sequence -- the connector is
    # stateless over the ``records=`` list.
    second_pass = list(source.iter_records())
    assert second_pass == records


# ===========================================================================
# Source 4: operator-supplied integral / derivation pairs
# (Requirement 3.1, bullet 4)
# ===========================================================================


@pytest.mark.integration
def test_operator_integral_source_smoke() -> None:
    """Smoke test for :class:`OperatorIntegralSource` (Requirement 3.1).

    Operator-curated integral and derivation pairs share the explicit
    ``Reasoning_Format`` shape with the open step-by-step corpus; the
    connector mirrors the passthrough behavior of
    :class:`OpenStepByStepSource` but defaults to the
    ``"user-provided"`` license so the dataset card surfaces the fact
    that license terms come from the operator's own agreement, not from
    an upstream public dataset.

    This test feeds 2 inline records (one indefinite integral, one
    symbolic derivation) and verifies:

    * Protocol conformance and the documented default ``source_id`` /
      ``license``.
    * Per-record passthrough of ``problem``, ``solution_steps``, and
      ``final_answer``.
    * LaTeX preservation in the integral problem (Requirement 3.6).
    * Round-trip stability across two iterations.
    """

    inline_records = [
        {
            "problem": "Compute the indefinite integral $\\int x^2 \\, dx$.",
            "solution_steps": [
                "Apply the power rule: \\int x^n \\, dx = x^{n+1}/(n+1) + C.",
                "With n=2 we obtain x^3/3 + C.",
            ],
            "final_answer": "x^3/3 + C",
        },
        {
            "problem": "Derive d/dx[x * cos(x)].",
            "solution_steps": [
                "Apply the product rule: (uv)' = u'v + uv'.",
                "u = x, v = cos(x); u' = 1, v' = -sin(x).",
                "Therefore d/dx[x * cos(x)] = cos(x) - x * sin(x).",
            ],
            "final_answer": "cos(x) - x*sin(x)",
        },
    ]
    source = OperatorIntegralSource(records=inline_records)

    # Step 1: interface availability and operator-specific defaults.
    assert isinstance(source, DatasetSource)
    assert source.source_id == "operator_integrals"
    # Operator-supplied data carries operator-governed terms; the
    # default license tag must reflect that distinction.
    assert source.license == "user-provided"
    assert source.record_count() == 2

    # Step 2: fixture round-trip.
    records = list(source.iter_records())
    assert len(records) == 2

    first = records[0]
    # Requirement 3.6: LaTeX delimiters and notation survive ingest.
    assert "$\\int x^2 \\, dx$" in first["problem"]
    assert first["solution_steps"] == [
        "Apply the power rule: \\int x^n \\, dx = x^{n+1}/(n+1) + C.",
        "With n=2 we obtain x^3/3 + C.",
    ]
    assert first["final_answer"] == "x^3/3 + C"

    second = records[1]
    assert second["problem"] == "Derive d/dx[x * cos(x)]."
    assert second["solution_steps"] is not None
    assert len(second["solution_steps"]) == 3
    assert second["final_answer"] == "cos(x) - x*sin(x)"

    # The connector must be stable across iterations so that
    # Dataset_Builder can iterate it during ingest counting and again
    # during normalization (task 4.3) without losing records.
    second_pass = list(source.iter_records())
    assert second_pass == records
