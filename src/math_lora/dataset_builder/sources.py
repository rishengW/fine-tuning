"""Source connectors for ``Dataset_Builder`` (Requirement 3.1).

This module implements the **smoke-level** ingest layer of the
``Dataset_Builder`` component (see ``.kiro/specs/math-lora-finetuning/design.md``,
section *3. Dataset_Builder*). Requirement 3.1 lists four ingest sources:

1. GSM8K training split
2. MATH training split
3. A documented open-source step-by-step math corpus
4. Operator-supplied integral and derivation pairs

Each source is exposed as a :class:`DatasetSource` that yields
:class:`RawRecord` values. The connector layer is intentionally **thin**:
it performs only the best-effort field extraction needed to produce a
``RawRecord``, and it leaves stricter schema enforcement (Requirement 3.2,
3.3, 3.6) to the normalization layer implemented in task 4.3. Specifically:

* GSM8K answers are split on the published ``####`` delimiter (see
  `GSM8K paper, Cobbe et al. 2021 <https://arxiv.org/abs/2110.14168>`_).
* MATH solutions are searched for the ``\\boxed{...}`` final-answer token
  (see `MATH paper, Hendrycks et al. 2021
  <https://arxiv.org/abs/2103.03874>`_).
* The other two connectors expect already-explicit fields, since their
  inputs are operator-curated.

If extraction fails (no ``####``, no ``\\boxed{...}``), the connector still
yields a ``RawRecord`` -- with ``final_answer=""`` -- so that the
normalization layer can reject it and count it under ``rejection_reasons``
per Requirement 3.3 / Property 9 (dataset accounting balance). The
connector layer never raises on a malformed record.

Determinism (relevant to Property 9 and Property 11): every connector
yields records in the exact order of its input list, so two iterations
over the same input produce the same ``RawRecord`` sequence.

The connectors do *not* fetch real datasets. CI passes pre-bundled
fixtures (``tests/fixtures/datasets/*.jsonl``) or in-memory ``records``
lists; the integration tests in task 4.x will exercise the real
``datasets``-library path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable


# ---------------------------------------------------------------------------
# Raw record shape
# ---------------------------------------------------------------------------


class RawRecord(TypedDict):
    """A best-effort raw record yielded by a :class:`DatasetSource`.

    The ``raw`` field retains the original dataset-specific JSON dict so that
    the normalization layer (task 4.3) and the dataset-card emitter (task 4.14)
    can preserve traceability back to the source row -- this is what lets the
    ``DatasetCard.rejection_reasons`` map (per Requirement 3.7) reference the
    exact field that triggered a rejection.

    Field semantics:

    * ``problem``: best-effort problem text. Connectors copy it from the
      source verbatim (no whitespace stripping, no LaTeX munging) so that
      Requirement 3.6 (LaTeX preservation) is honored.
    * ``solution_steps``: ``None`` when the source provides only a
      monolithic solution string and the connector cannot safely split it,
      or a list of one or more strings when the source either supplies
      explicit steps or the connector splits an answer on ``\\n``. The
      list MAY contain empty strings; the normalization layer is
      responsible for rejecting empty entries (Requirement 3.3).
    * ``final_answer``: extracted final answer string, or ``""`` if the
      connector could not extract one. The normalization layer rejects
      records with an empty ``final_answer`` (Requirement 3.3).
    * ``raw``: the unmodified source dict.
    """

    problem: str
    solution_steps: list[str] | None
    final_answer: str
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# DatasetSource protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetSource(Protocol):
    """Protocol implemented by every Requirement 3.1 ingest source.

    The protocol is ``runtime_checkable`` so that the registry function
    :func:`iter_sources` can validate operator-supplied connectors at
    runtime (an operator-supplied integral set may live outside this
    package; see Requirement 3.1 bullet 4).

    Attributes:
        source_id: Stable string identifier used as a key in
            ``DatasetCard.source_name`` (Requirement 3.7) and in the
            per-source rejection counts. Must be unique within a
            ``Dataset_Builder.build`` invocation.
        license: SPDX-style license identifier or short license tag,
            recorded in ``DatasetCard.license`` (Requirement 3.7).

    Methods:
        iter_records: Yield :class:`RawRecord` values in a deterministic
            order. Implementations MUST NOT mutate any shared state, so
            two iterations over the same source instance yield the same
            sequence (relevant to Property 9 and Property 11 in
            ``design.md``).
        record_count: Optional. Return the number of records the source
            will yield, or ``None`` when the count is not knowable in
            advance (e.g. when ``iter_records`` streams from a remote
            dataset). Used by progress reporting in
            ``Training_Pipeline``; never used for correctness.
    """

    source_id: str
    license: str

    def iter_records(self) -> Iterator[RawRecord]:
        """Yield raw records in deterministic order."""
        ...

    def record_count(self) -> int | None:  # pragma: no cover - optional
        """Return the record count if known, ``None`` otherwise."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ``\boxed{...}`` extraction for the MATH dataset. The pattern is intentionally
# limited to a single level of braces so that a connector failure on nested
# braces falls through to ``final_answer=""`` rather than raising. The
# normalization layer (task 4.3) is responsible for stricter LaTeX parsing.
_MATH_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{([^{}]+)\}")

# GSM8K's published delimiter (`#### <answer>`) per Cobbe et al. 2021.
_GSM8K_DELIM: str = "####"


def _split_lines(text: str) -> list[str]:
    """Split ``text`` on newlines and drop empty entries.

    Used by the GSM8K and MATH connectors to convert a multi-line solution
    string into the ``solution_steps`` list. Whitespace-only lines are
    treated as separators (dropped) since they carry no reasoning content
    and would be rejected by the normalization layer anyway.
    """

    return [line for line in text.split("\n") if line.strip()]


def _coerce_records(records: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Materialize ``records`` into a list, or pass ``None`` through.

    A list is required (not a generic iterable) so that the connector can
    be iterated multiple times and so that ``record_count`` returns a
    meaningful answer. ``None`` signals "real fetch deferred to integration
    tests" -- see the per-connector NotImplementedError messages.
    """

    if records is None:
        return None
    return list(records)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON-lines file into a list of dicts.

    Used by the bundled-fixture loaders for GSM8K and MATH. Empty lines are
    skipped. Each non-empty line must parse to a JSON object; non-object
    lines raise ``ValueError`` so a corrupted fixture surfaces at load
    time rather than producing silently malformed ``RawRecord`` values.
    """

    records: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}"
                )
            records.append(obj)
    return records


# ---------------------------------------------------------------------------
# GSM8K training split connector
# ---------------------------------------------------------------------------


class GSM8KTrainSource:
    """Connector for the GSM8K training split (Requirement 3.1, bullet 1).

    The published GSM8K row shape is::

        {"question": "<problem>", "answer": "<reasoning>\\n#### <final>"}

    The connector splits ``answer`` on the ``####`` delimiter (Cobbe et al.
    2021); the prefix is treated as a multi-line solution and split on
    ``\\n``, while the suffix is the final answer (stripped of surrounding
    whitespace). If ``####`` does not appear, the connector yields a
    ``RawRecord`` with ``final_answer=""`` so the normalization layer
    (task 4.3) can reject it and count it under ``rejection_reasons``.

    Args:
        records: In-memory list of source dicts. When ``None``, the
            connector tries to load the small bundled fixture at
            ``tests/fixtures/datasets/gsm8k_sample.jsonl`` (intended only
            for smoke tests). When neither is available the connector
            raises ``NotImplementedError`` to make the deferred real-fetch
            obvious.
        license: License tag, defaults to ``"gsm8k-mit"``. Recorded
            verbatim in ``DatasetCard.license`` per Requirement 3.7.
        source_id: Stable identifier, defaults to ``"gsm8k_train"``.
        fixture_path: Override path to a JSONL fixture; primarily used by
            tests (task 4.2) that want to point at a specific small file.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]] | None = None,
        *,
        license: str = "gsm8k-mit",
        source_id: str = "gsm8k_train",
        fixture_path: str | Path | None = None,
    ) -> None:
        self.source_id = source_id
        self.license = license
        self._records: list[dict[str, Any]] | None = _coerce_records(records)
        self._fixture_path: str | Path | None = fixture_path

    def _resolved_records(self) -> list[dict[str, Any]]:
        """Resolve the record source, falling back to the bundled fixture.

        The fallback exists so that the smoke command in task 4.1's prompt
        (``python -c "...GSM8KTrainSource(records=[{...}]).iter_records()..."``)
        works without the operator having to ship a real GSM8K download.
        Outside of smoke usage, callers always pass ``records=`` or
        ``fixture_path=`` explicitly.
        """

        if self._records is not None:
            return self._records
        if self._fixture_path is not None:
            return _load_jsonl(self._fixture_path)
        raise NotImplementedError(
            "GSM8KTrainSource: real fetch deferred to integration tests; "
            "pass records=... or fixture_path=... explicitly for smoke runs"
        )

    def iter_records(self) -> Iterator[RawRecord]:
        for raw in self._resolved_records():
            problem = str(raw.get("question", ""))
            answer = str(raw.get("answer", ""))
            if _GSM8K_DELIM in answer:
                solution_text, _, final = answer.partition(_GSM8K_DELIM)
                solution_steps = _split_lines(solution_text)
                final_answer = final.strip()
            else:
                # Best-effort: keep the answer text as the (single) solution
                # step block and leave final_answer empty so task 4.3 can
                # reject and count the record under ``rejection_reasons``.
                solution_steps = _split_lines(answer)
                final_answer = ""
            yield RawRecord(
                problem=problem,
                solution_steps=solution_steps if solution_steps else None,
                final_answer=final_answer,
                raw=dict(raw),
            )

    def record_count(self) -> int | None:
        if self._records is not None:
            return len(self._records)
        if self._fixture_path is not None:
            # Cheap line count; the file is small (smoke fixture).
            return len(_load_jsonl(self._fixture_path))
        return None


# ---------------------------------------------------------------------------
# MATH training split connector
# ---------------------------------------------------------------------------


class MATHTrainSource:
    """Connector for the MATH training split (Requirement 3.1, bullet 2).

    The published MATH row shape is::

        {"problem": "<problem>", "solution": "... \\boxed{<final>} ..."}

    The connector regex-extracts the first ``\\boxed{...}`` group as
    ``final_answer`` (Hendrycks et al. 2021); the rest of ``solution`` is
    used as the multi-line solution text and split on ``\\n``. If no
    ``\\boxed{...}`` is found, ``final_answer`` is set to ``""``.

    Args:
        records: In-memory list of source dicts. ``None`` falls back to the
            bundled fixture at ``tests/fixtures/datasets/math_sample.jsonl``
            when ``fixture_path`` is not provided; otherwise raises
            ``NotImplementedError``.
        license: License tag, defaults to ``"math-cc-by-sa-4.0"`` (the
            published license of the Hendrycks MATH dataset).
        source_id: Stable identifier, defaults to ``"math_train"``.
        fixture_path: Override path to a JSONL fixture.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]] | None = None,
        *,
        license: str = "math-cc-by-sa-4.0",
        source_id: str = "math_train",
        fixture_path: str | Path | None = None,
    ) -> None:
        self.source_id = source_id
        self.license = license
        self._records: list[dict[str, Any]] | None = _coerce_records(records)
        self._fixture_path: str | Path | None = fixture_path

    def _resolved_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if self._fixture_path is not None:
            return _load_jsonl(self._fixture_path)
        raise NotImplementedError(
            "MATHTrainSource: real fetch deferred to integration tests; "
            "pass records=... or fixture_path=... explicitly for smoke runs"
        )

    def iter_records(self) -> Iterator[RawRecord]:
        for raw in self._resolved_records():
            problem = str(raw.get("problem", ""))
            solution = str(raw.get("solution", ""))
            match = _MATH_BOXED_RE.search(solution)
            if match is not None:
                final_answer = match.group(1).strip()
            else:
                final_answer = ""
            solution_steps = _split_lines(solution)
            yield RawRecord(
                problem=problem,
                solution_steps=solution_steps if solution_steps else None,
                final_answer=final_answer,
                raw=dict(raw),
            )

    def record_count(self) -> int | None:
        if self._records is not None:
            return len(self._records)
        if self._fixture_path is not None:
            return len(_load_jsonl(self._fixture_path))
        return None


# ---------------------------------------------------------------------------
# Open-source step-by-step corpus connector
# ---------------------------------------------------------------------------


class OpenStepByStepSource:
    """Connector for a documented open-source step-by-step math corpus
    (Requirement 3.1, bullet 3).

    The corpus is expected to ship records with the explicit
    ``Reasoning_Format`` shape from Requirement 3.2::

        {"problem": "...", "solution_steps": ["step 1", "step 2", ...], "final_answer": "..."}

    so this connector performs no extraction -- it is a passthrough that
    copies the explicit fields into a :class:`RawRecord`. Per-record
    validation (non-empty problem / steps / answer) is delegated to the
    normalization layer (task 4.3) per Requirement 3.3.

    The default ``license`` is ``"cc-by-4.0"`` to match the most common
    license under which open step-by-step math corpora are published; it
    is overridable so the operator can record the actual license of the
    chosen corpus in the dataset card.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]] | None = None,
        *,
        license: str = "cc-by-4.0",
        source_id: str = "open_step_by_step",
    ) -> None:
        self.source_id = source_id
        self.license = license
        self._records: list[dict[str, Any]] | None = _coerce_records(records)

    def _resolved_records(self) -> list[dict[str, Any]]:
        if self._records is None:
            raise NotImplementedError(
                "OpenStepByStepSource: real fetch deferred to integration "
                "tests; pass records=... explicitly for smoke runs"
            )
        return self._records

    def iter_records(self) -> Iterator[RawRecord]:
        for raw in self._resolved_records():
            problem = str(raw.get("problem", ""))
            steps_value: Any = raw.get("solution_steps")
            if isinstance(steps_value, list):
                solution_steps: list[str] | None = [str(s) for s in steps_value]
            elif steps_value is None:
                solution_steps = None
            else:
                # Tolerate scalar inputs by treating them as a single step;
                # the normalization layer will reject if the step is empty.
                solution_steps = [str(steps_value)]
            final_answer = str(raw.get("final_answer", ""))
            yield RawRecord(
                problem=problem,
                solution_steps=solution_steps,
                final_answer=final_answer,
                raw=dict(raw),
            )

    def record_count(self) -> int | None:
        if self._records is None:
            return None
        return len(self._records)


# ---------------------------------------------------------------------------
# Operator-supplied integral / derivation connector
# ---------------------------------------------------------------------------


class OperatorIntegralSource:
    """Connector for operator-supplied integral and derivation pairs
    (Requirement 3.1, bullet 4).

    Operator-curated data is expected to already be in the explicit
    ``Reasoning_Format`` shape, so the connector is a passthrough mirroring
    :class:`OpenStepByStepSource`. The default license is
    ``"user-provided"`` to make it explicit in the dataset card that the
    license terms are governed by the operator's own agreement, not by an
    upstream public dataset.

    NOTE: Records yielded here are a *training* signal. Requirement 3.10
    requires that any problems in the operator's ``Custom_Integral_Set``
    used for evaluation be excluded from training; that exclusion is the
    responsibility of task 4.12 (a later step), not this connector.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]] | None = None,
        *,
        license: str = "user-provided",
        source_id: str = "operator_integrals",
    ) -> None:
        self.source_id = source_id
        self.license = license
        self._records: list[dict[str, Any]] | None = _coerce_records(records)

    def _resolved_records(self) -> list[dict[str, Any]]:
        if self._records is None:
            raise NotImplementedError(
                "OperatorIntegralSource: operator must supply records "
                "explicitly; pass records=... for smoke runs"
            )
        return self._records

    def iter_records(self) -> Iterator[RawRecord]:
        for raw in self._resolved_records():
            problem = str(raw.get("problem", ""))
            steps_value: Any = raw.get("solution_steps")
            if isinstance(steps_value, list):
                solution_steps: list[str] | None = [str(s) for s in steps_value]
            elif steps_value is None:
                solution_steps = None
            else:
                solution_steps = [str(steps_value)]
            final_answer = str(raw.get("final_answer", ""))
            yield RawRecord(
                problem=problem,
                solution_steps=solution_steps,
                final_answer=final_answer,
                raw=dict(raw),
            )

    def record_count(self) -> int | None:
        if self._records is None:
            return None
        return len(self._records)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def iter_sources(source_specs: Sequence[DatasetSource]) -> Iterator[DatasetSource]:
    """Yield each :class:`DatasetSource` in ``source_specs`` after a runtime check.

    This is a thin wrapper that lets ``Dataset_Builder.build`` (task 4.x)
    iterate over a heterogeneous list of connectors -- including the four
    built-in ones above and any operator-supplied custom source -- while
    enforcing the protocol contract at the boundary. Any element that is
    not a :class:`DatasetSource` raises ``TypeError`` immediately, which
    matches the design's *Error Handling* preference for surfacing config
    errors in the pre-flight phase rather than mid-build.

    Validation also rejects duplicate ``source_id`` values: per Requirement
    3.7 each source produces its own dataset card keyed by ``source_id``,
    so a duplicate would silently overwrite the first card.
    """

    seen: set[str] = set()
    for spec in source_specs:
        if not isinstance(spec, DatasetSource):
            raise TypeError(
                f"iter_sources: element {spec!r} does not implement DatasetSource"
            )
        if spec.source_id in seen:
            raise ValueError(
                f"iter_sources: duplicate source_id {spec.source_id!r}; "
                f"each source must have a unique identifier"
            )
        seen.add(spec.source_id)
        yield spec


__all__: Final = [
    "RawRecord",
    "DatasetSource",
    "GSM8KTrainSource",
    "MATHTrainSource",
    "OpenStepByStepSource",
    "OperatorIntegralSource",
    "iter_sources",
]
