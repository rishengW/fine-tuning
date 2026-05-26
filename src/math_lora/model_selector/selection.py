"""Eligibility, feasibility, and scoring primitives for ``Model_Selector``.

This module implements the three selection-stage primitives required by
task 2.3 of the implementation plan:

1. **Eligibility check** (Requirement 1.10) -- given a raw candidate
   payload (a mapping such as a YAML/JSON dict), report whether the
   payload carries every field required by Requirements 1.1 and 1.8 and,
   if not, list the absent field names. A candidate that is ineligible is
   excluded from ranking by the orchestrator (task 2.4).
2. **Feasibility flag and VRAM shortfall** (Requirement 1.4) -- given the
   estimated minimum VRAM for an eligible candidate and the
   :class:`HardwareProfile` VRAM, return ``(feasible, shortfall_gb)``
   where ``shortfall_gb`` is rounded to one decimal place when the
   candidate is infeasible and ``None`` otherwise.
3. **Scoring function** (Requirement 1.5) -- compute a deterministic
   weighted score in ``[0.0, 1.0]`` from the four documented inputs (GSM8K
   baseline, MATH baseline, parameter count, license permissiveness) using
   weights that sum to ``1.0``, and pick a winner from a list of scored
   candidates using a lexicographic tie-breaker on ``model_id``.

The functions in this module are **pure**: they never read time, random
state, environment variables, or any other ambient input. Identical inputs
produce identical outputs, which is part of the bit-for-bit reproducibility
guarantee in Requirement 8.10 and the determinism leg of design Property 2
(selection-report soundness).

References:
    * ``.kiro/specs/math-lora-finetuning/requirements.md`` -- Requirement 1
      (acceptance criteria 1.1, 1.4, 1.5, 1.8, 1.10).
    * ``.kiro/specs/math-lora-finetuning/design.md`` --
      § Components / 1. ``Model_Selector`` (scoring formula and
      tie-breaker rule), § Data Models / ``CandidateEntry`` (output shape
      consumed by task 2.4 when assembling the ``SelectionReport``),
      Property 2 (selection report soundness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence


# ---------------------------------------------------------------------------
# Required-field tables (Requirements 1.1, 1.8)
# ---------------------------------------------------------------------------


#: Fields required by Requirement 1.1 ("each candidate declares its parameter
#: count in billions, license identifier, native context length in tokens,
#: tokenizer family, a baseline GSM8K score..., and a baseline MATH score...").
#:
#: ``model_id`` is also required because the eligibility report keys
#: per-candidate entries by ``model_id`` -- a candidate whose identifier is
#: missing cannot be reported back to the operator at all.
_REQUIRED_FIELDS_REQ_1_1: Final[tuple[str, ...]] = (
    "model_id",
    "param_count_b",
    "license_id",
    "native_context_length_tokens",
    "tokenizer_family",
    "baseline_gsm8k",
    "baseline_math",
)

#: Fields required by Requirement 1.8 ("THE Model_Selector SHALL record, for
#: each candidate Base_Model, three boolean flags indicating whether the
#: candidate's published license file permits fine-tuning, redistribution
#: of Adapters, and commercial use, together with the license identifier").
#:
#: ``license_id`` is intentionally listed under both 1.1 and 1.8 (the spec
#: says "together with the license identifier") -- the deduplication
#: happens in :data:`REQUIRED_FIELDS` below so it appears only once in
#: ``missing_fields`` even when both clauses are missing it.
_REQUIRED_FIELDS_REQ_1_8: Final[tuple[str, ...]] = (
    "license_id",
    "license_allows_finetuning",
    "license_allows_adapter_redistribution",
    "license_allows_commercial_use",
)


def _dedupe_preserving_order(items: Sequence[str]) -> tuple[str, ...]:
    """Return ``items`` with duplicates removed, preserving first-seen order.

    The eligibility report names the fields in the order they appear in the
    requirement document, so we cannot simply ``sorted(set(...))``.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


#: Union of the fields required by 1.1 and 1.8, deduplicated and ordered.
#: Exposed as a module-level constant so callers (task 2.4 and tests) can
#: enumerate the required-field list without re-deriving it.
REQUIRED_FIELDS: Final[tuple[str, ...]] = _dedupe_preserving_order(
    (*_REQUIRED_FIELDS_REQ_1_1, *_REQUIRED_FIELDS_REQ_1_8)
)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityResult:
    """Eligibility verdict for one raw candidate payload (Requirement 1.10).

    Attributes:
        eligible: ``True`` iff every field listed in :data:`REQUIRED_FIELDS`
            is present in the input payload. Eligibility checks only
            *presence*, not type or value validity -- type/value
            violations are caught by :class:`BaseModelCandidate.parse` in
            :mod:`math_lora.types.models` and surface as
            :class:`SchemaValidationError` rather than as ineligibility.
        missing_fields: Tuple of absent field names in declaration order
            (the order they appear in :data:`REQUIRED_FIELDS`). Empty
            tuple iff ``eligible`` is ``True``.

    Rationale for splitting presence-checking from full schema validation:
    Requirement 1.10 explicitly requires that the *Model_Selector* exclude
    such candidates from ranking and *list the missing field names in the
    selection report*. If we used pydantic's ``parse`` to detect missing
    fields, we would also fail-loud on unrelated value violations (e.g.,
    ``param_count_b == 0``), which would be a stricter behaviour than
    Requirement 1.10 mandates and would prevent the selector from
    reporting the offending candidate at all. So eligibility is a strict
    subset of schema validity: missing field -> ineligible (still
    reported), invalid field value -> the orchestrator decides whether to
    surface it or silently skip (out of scope for task 2.3).
    """

    eligible: bool
    missing_fields: tuple[str, ...] = ()


def _is_present(value: Any) -> bool:
    """Return ``True`` iff ``value`` is considered present.

    A value is *absent* when it is ``None``. Empty strings and zero
    numbers are considered present here; the schema layer
    (:class:`BaseModelCandidate.parse`) is responsible for rejecting them
    on value grounds. Splitting "absent" from "invalid" lets the selector
    report ineligibility (missing fields) separately from schema rejection
    (invalid values), as the design's *Error Handling* table requires.
    """

    return value is not None


def check_eligibility(payload: Mapping[str, Any]) -> EligibilityResult:
    """Check whether ``payload`` carries every Req 1.1 / 1.8 field.

    Args:
        payload: A raw candidate payload, typically a parsed YAML/JSON
            dict. The function does not require the payload to be a
            :class:`BaseModelCandidate` instance -- the whole point of
            eligibility is to handle inputs that would *fail* pydantic
            validation because of missing fields.

    Returns:
        :class:`EligibilityResult` with ``eligible == True`` iff every
        field in :data:`REQUIRED_FIELDS` is present in ``payload`` (and
        not ``None``). Otherwise ``eligible == False`` and
        ``missing_fields`` lists the absent field names in declaration
        order.

    Determinism:
        Pure function. Identical input mappings produce identical
        :class:`EligibilityResult` values, which Property 2(d) requires.
    """

    missing: list[str] = []
    for name in REQUIRED_FIELDS:
        if name not in payload or not _is_present(payload[name]):
            missing.append(name)

    return EligibilityResult(
        eligible=not missing,
        missing_fields=tuple(missing),
    )


# ---------------------------------------------------------------------------
# Feasibility (Requirement 1.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityResult:
    """Feasibility verdict for one eligible candidate (Requirement 1.4).

    Attributes:
        feasible: ``True`` iff the candidate's estimated minimum VRAM is
            no greater than the operator's available VRAM per GPU.
            Property 2(b) pins the equivalence ``feasible ==
            (estimated_min_vram_gb <= hw.vram_per_gpu_gb)``.
        vram_shortfall_gb: ``None`` when ``feasible`` is ``True``;
            otherwise the gap rounded to one decimal place per
            Requirement 1.4 ("SHALL record the shortfall in gigabytes
            rounded to one decimal place"). The shortfall value satisfies
            Property 2(c): ``vram_shortfall_gb ==
            round(estimated_min_vram_gb - hw.vram_per_gpu_gb, 1)``.
    """

    feasible: bool
    vram_shortfall_gb: float | None = None


def check_feasibility(
    estimated_min_vram_gb: float,
    available_vram_gb: float,
) -> FeasibilityResult:
    """Compute the feasibility flag and VRAM shortfall.

    Args:
        estimated_min_vram_gb: The candidate's estimated minimum VRAM in
            gigabytes, typically produced by
            :func:`math_lora.model_selector.estimate_min_vram_gb`.
        available_vram_gb: The operator's available VRAM per GPU in
            gigabytes, typically taken from
            :attr:`HardwareProfile.vram_per_gpu_gb`. Accepted as ``float``
            (rather than ``int``) so callers that scale VRAM by reserved-
            overhead fractions can pass a non-integer value without
            artificial rounding here.

    Returns:
        :class:`FeasibilityResult`. When the candidate is feasible the
        shortfall field is ``None``; when infeasible the shortfall is
        ``round(estimated - available, 1)`` per Requirement 1.4. The
        shortfall is **strictly positive** in the infeasible case (a zero
        shortfall would mean ``feasible == True``).

    Determinism:
        Pure function. The Python built-in :func:`round` uses banker's
        rounding, which is bit-deterministic on a fixed CPython build, so
        the result is reproducible across runs (a precondition for
        Requirement 8.10).
    """

    feasible = estimated_min_vram_gb <= available_vram_gb
    if feasible:
        return FeasibilityResult(feasible=True, vram_shortfall_gb=None)
    # Round to one decimal place per Requirement 1.4. Subtracting and then
    # rounding (rather than rounding each operand first) keeps the
    # reported gap honest even when both operands have many fractional
    # digits.
    shortfall = round(estimated_min_vram_gb - available_vram_gb, 1)
    return FeasibilityResult(feasible=False, vram_shortfall_gb=shortfall)


# ---------------------------------------------------------------------------
# Scoring (Requirement 1.5)
# ---------------------------------------------------------------------------


# Reasonable default weights summing to 1.0. The scoring function accepts
# any caller-supplied weights that sum to 1.0 (see :class:`ScoringWeights`),
# but the orchestrator at task 2.4 uses these defaults so that the emitted
# selection report has a consistent shape across invocations.
#
# Rationale for the default split:
# - Math benchmarks (GSM8K + MATH) carry the bulk of the signal because the
#   feature is *math* fine-tuning; together they receive 0.7.
# - Parameter count is a weak negative signal at selection time (smaller
#   models are cheaper to train and serve) but a positive signal for
#   capability ceiling. We weight it modestly at 0.1 and ``normalize`` it
#   by inverting the [0.5B, 70B] band to a [0,1] score in
#   :func:`_normalize_param_count_b` -- bigger models => higher normalized
#   value, which the scoring function then rewards.
# - License permissiveness gets the remaining 0.2: a fully-permissive
#   license that allows fine-tuning, redistribution, and commercial use is
#   strictly better than a restrictive one for a single-student researcher
#   who may want to publish artifacts.
_DEFAULT_W_GSM: Final[float] = 0.40
_DEFAULT_W_MATH: Final[float] = 0.30
_DEFAULT_W_PARAMS: Final[float] = 0.10
_DEFAULT_W_LICENSE: Final[float] = 0.20

#: Param-count band used by the default normalizer. The lower bound is
#: ``0.5`` so that a 0.5B model normalizes to 0.0 and the upper bound is
#: ``70.0`` so that a 70B model normalizes to 1.0. Models outside the band
#: clamp to the nearest endpoint.
_PARAMS_NORMALIZE_LO_B: Final[float] = 0.5
_PARAMS_NORMALIZE_HI_B: Final[float] = 70.0

#: Tolerance used when checking that scoring weights sum to 1.0. Small
#: enough to catch obvious typos (0.39 + 0.30 + 0.10 + 0.20 = 0.99) but
#: large enough to ignore floating-point noise from honest decompositions.
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the scoring function (Requirement 1.5).

    Attributes:
        w_gsm: Weight on the normalized GSM8K baseline score.
        w_math: Weight on the normalized MATH baseline score.
        w_params: Weight on the normalized parameter count.
        w_license: Weight on the license-permissiveness value.

    Constraint:
        ``w_gsm + w_math + w_params + w_license == 1.0`` (within
        :data:`_WEIGHT_SUM_TOLERANCE`). The constructor validates this so
        that the orchestrator at task 2.4 can record the weights in the
        selection report verbatim and the consumer of the report can
        re-derive the score byte-for-byte.

    Why a closed-form weights structure rather than ``**kwargs``? Because
    Requirement 1.6 requires that the report list "the weights used in the
    scoring function". A typed dataclass makes the four fields explicit
    and lets us serialize them with stable field names.
    """

    w_gsm: float = _DEFAULT_W_GSM
    w_math: float = _DEFAULT_W_MATH
    w_params: float = _DEFAULT_W_PARAMS
    w_license: float = _DEFAULT_W_LICENSE

    def __post_init__(self) -> None:
        # Each weight individually must be in [0, 1] -- a negative weight
        # would invert the contribution sign and break the
        # ``score in [0, 1]`` claim of Property 2(f); a weight > 1 would
        # also break it because the inputs are already in [0, 1].
        for name in ("w_gsm", "w_math", "w_params", "w_license"):
            value = getattr(self, name)
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"ScoringWeights.{name} must be in [0.0, 1.0], got {value!r}"
                )

        total = self.w_gsm + self.w_math + self.w_params + self.w_license
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "ScoringWeights must sum to 1.0 "
                f"(got w_gsm={self.w_gsm}, w_math={self.w_math}, "
                f"w_params={self.w_params}, w_license={self.w_license}, "
                f"sum={total})"
            )


# ---- normalization helpers (Requirement 1.5) ------------------------------


def _normalize_baseline_score(value: float) -> float:
    """Clamp a benchmark accuracy in ``[0, 1]`` and pass it through.

    GSM8K and MATH baselines are already declared in ``[0.0, 1.0]`` per
    Requirement 1.1, so the "normalize" step is a no-op when the input is
    in range. We still clamp explicitly so the function is total over the
    real line and so that minor float noise (e.g. ``0.9999999999999999``
    due to YAML round-tripping) does not push the score above 1.0.
    """

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _normalize_param_count_b(param_count_b: float) -> float:
    """Map a parameter count in billions onto ``[0, 1]``.

    Linear scaling between :data:`_PARAMS_NORMALIZE_LO_B` (0.5B -> 0.0)
    and :data:`_PARAMS_NORMALIZE_HI_B` (70B -> 1.0). Values outside the
    band clamp to the nearest endpoint. The band covers the candidate
    families called out in the design (Qwen2.5-Math 1.5B/7B, DeepSeek-Math
    7B, Llama-3 70B) plus a healthy margin on either side.

    Bigger models receive a higher normalized value, which the scoring
    function then rewards via ``w_params``. This matches the spec's
    intuition that "stronger math priors and bigger context budgets are
    better -- if they fit". Whether they fit is decided by the feasibility
    check, not the scoring function: an infeasible candidate is never
    scored at all (its ``score`` is ``None`` per the design's
    ``CandidateEntry`` shape).
    """

    if param_count_b <= _PARAMS_NORMALIZE_LO_B:
        return 0.0
    if param_count_b >= _PARAMS_NORMALIZE_HI_B:
        return 1.0
    span = _PARAMS_NORMALIZE_HI_B - _PARAMS_NORMALIZE_LO_B
    return (float(param_count_b) - _PARAMS_NORMALIZE_LO_B) / span


def license_permissiveness(
    allows_finetuning: bool,
    allows_adapter_redistribution: bool,
    allows_commercial_use: bool,
) -> float:
    """Compute a license-permissiveness value in ``[0, 1]``.

    The value is the unweighted mean of the three flags, treated as
    ``1.0`` when ``True`` and ``0.0`` when ``False``. A fully-permissive
    license (all three flags ``True``) scores ``1.0``; a fully-restrictive
    license scores ``0.0``; mixed cases land at thirds.

    Why a simple mean? Requirement 1.5 says the value is "derived from
    whether the license permits fine-tuning, redistribution of Adapters,
    and commercial use" but does not pin the derivation. The mean is the
    simplest commutative aggregator that maps the boolean cube to
    ``{0, 1/3, 2/3, 1}`` and is the value recorded in the selection
    report (Requirement 1.6).
    """

    flags = (allows_finetuning, allows_adapter_redistribution, allows_commercial_use)
    return sum(1.0 for flag in flags if flag) / len(flags)


# ---- core scoring ---------------------------------------------------------


@dataclass(frozen=True)
class ScoringInputs:
    """The four inputs to :func:`compute_score` (Requirement 1.5).

    Bundled as a frozen dataclass so the orchestrator at task 2.4 can pass
    one structured value rather than four positional floats and so that
    later code (the selection report assembler) can attach this dataclass
    verbatim to each :class:`CandidateEntry` for auditing.
    """

    baseline_gsm8k: float
    baseline_math: float
    param_count_b: float
    license_permissiveness: float


def compute_score(
    inputs: ScoringInputs,
    weights: ScoringWeights | None = None,
) -> float:
    """Compute the deterministic score for an eligible feasible candidate.

    Implements the formula in Requirement 1.5 and the design document's
    *Components and Interfaces* section / 1. ``Model_Selector``::

        score = w_gsm * normalize(gsm8k_baseline)
              + w_math * normalize(math_baseline)
              + w_params * normalize(param_count_b)
              + w_license * license_permissiveness

    Args:
        inputs: The four scoring inputs. Each field is validated by being
            run through the corresponding normalizer (which clamps into
            ``[0, 1]``).
        weights: Optional override of the default :class:`ScoringWeights`.
            When ``None``, a fresh default :class:`ScoringWeights` is
            constructed. The weights MUST sum to ``1.0`` (the constructor
            of :class:`ScoringWeights` enforces this).

    Returns:
        A real number in ``[0.0, 1.0]``. Property 2(f) requires every
        score on the report to lie in this closed interval; the function
        guarantees that property structurally because (i) every
        normalized input is in ``[0, 1]``, (ii) the weights are
        non-negative, and (iii) the weights sum to ``1``, so the result
        is a convex combination of values in ``[0, 1]``.

    Determinism:
        Pure function. Repeated calls with equal inputs return the same
        bit-pattern float (modulo platform IEEE-754 conformance), which is
        a precondition for Property 2(e) ("``chosen_model_id`` ... refers
        to ... the maximum score").
    """

    w = weights if weights is not None else ScoringWeights()

    gsm = _normalize_baseline_score(inputs.baseline_gsm8k)
    math_score = _normalize_baseline_score(inputs.baseline_math)
    params = _normalize_param_count_b(inputs.param_count_b)
    lic = _normalize_baseline_score(inputs.license_permissiveness)

    return (
        w.w_gsm * gsm
        + w.w_math * math_score
        + w.w_params * params
        + w.w_license * lic
    )


# ---- ranking and tie-breaking --------------------------------------------


@dataclass(frozen=True)
class ScoredCandidate:
    """One ``(model_id, score)`` pair used by :func:`pick_winner`.

    A separate dataclass (rather than a tuple) so the tie-breaker rule
    reads clearly at the call site and so future fields (e.g., a
    confidence interval) can be added without breaking the API.
    """

    model_id: str
    score: float


def pick_winner(scored: Sequence[ScoredCandidate]) -> ScoredCandidate | None:
    """Pick the highest-scoring candidate; tie-break lexicographically on ``model_id``.

    Args:
        scored: A sequence of :class:`ScoredCandidate`. May be empty.

    Returns:
        The :class:`ScoredCandidate` with the maximum ``score``. When two
        or more candidates share the maximum, the one whose ``model_id``
        is **lexicographically smallest** wins. Returns ``None`` when
        ``scored`` is empty.

    Why a lexicographic tie-breaker on ``model_id``?
    Requirement 1.5 mandates that "the tie-breaker rule used when two
    candidates produce identical scores" be listed in the selection
    report. The design's *Components / 1. Model_Selector* section pins
    this to "lexicographic on ``model_id``", which is total, deterministic,
    and independent of insertion order -- so two runs that score the same
    candidates pick the same winner regardless of how the candidate list
    was assembled.

    Determinism:
        Pure function. Same input sequence -> same returned candidate.
    """

    if not scored:
        return None
    # Negate ``score`` for descending sort while keeping ``model_id``
    # ascending. Python's sort is stable, but we want the result to be
    # independent of the input order entirely, so we sort explicitly on
    # both keys rather than relying on stability.
    ranked = sorted(scored, key=lambda c: (-c.score, c.model_id))
    return ranked[0]


# ---------------------------------------------------------------------------
# CandidateEntry shape (consumed by the orchestrator at task 2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEntry:
    """One row of the selection report (design § Data Models / ``CandidateEntry``).

    This is the public output shape that :func:`evaluate_candidate` produces
    and that the task 2.4 orchestrator assembles into a ``SelectionReport``.
    Field semantics match the design's data-model block:

    Attributes:
        model_id: Stable identifier for the candidate. Always present
            (even for ineligible candidates, otherwise we could not key
            the report row by it).
        declared_param_count_b: The candidate's declared parameter count
            in billions, copied from the input payload. ``None`` only
            when the candidate is ineligible because of a missing
            ``param_count_b`` field.
        estimated_min_vram_gb: VRAM estimate from
            :func:`estimate_min_vram_gb`. ``None`` for ineligible
            candidates because the estimate has not been computed.
        feasible: ``True`` iff the candidate is eligible *and* its
            estimated VRAM does not exceed the available VRAM.
            Ineligible candidates have ``feasible == False`` because they
            were not evaluated.
        vram_shortfall_gb: One-decimal-place VRAM gap when infeasible;
            ``None`` otherwise. Property 2(c) ties this to a ``round``
            call.
        score: Numeric score in ``[0, 1]`` for eligible feasible
            candidates; ``None`` otherwise. Property 2(f) requires every
            non-null score to lie in the closed interval.
        license_allows_finetuning: Boolean flag from Requirement 1.8.
            Defaults to ``False`` for ineligible candidates whose payload
            did not declare it (so the report row is total).
        license_allows_adapter_redistribution: Boolean flag from
            Requirement 1.8. Same default rationale.
        license_allows_commercial_use: Boolean flag from Requirement 1.8.
            Same default rationale.
        license_id: License identifier from Requirement 1.8. ``None`` for
            ineligible candidates whose payload did not declare it.
        eligible: ``True`` iff every Requirement 1.1/1.8 field is
            present.
        missing_fields: List of absent field names for ineligible
            candidates; empty for eligible ones. Property 2(d) ties this
            to the per-payload eligibility check.
    """

    model_id: str
    declared_param_count_b: float | None
    estimated_min_vram_gb: float | None
    feasible: bool
    vram_shortfall_gb: float | None
    score: float | None
    license_allows_finetuning: bool
    license_allows_adapter_redistribution: bool
    license_allows_commercial_use: bool
    license_id: str | None
    eligible: bool
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


def evaluate_candidate(
    payload: Mapping[str, Any],
    estimated_min_vram_gb: float | None,
    available_vram_gb: float,
    weights: ScoringWeights | None = None,
) -> CandidateEntry:
    """Build a :class:`CandidateEntry` for one raw candidate payload.

    Combines the eligibility, feasibility, and scoring primitives into a
    single per-candidate result. The orchestrator at task 2.4 calls this
    once per candidate and assembles the resulting list into a
    ``SelectionReport``.

    Args:
        payload: Raw candidate payload (typically a dict). Eligibility is
            checked on this mapping directly; scoring, when applicable,
            reads the same fields.
        estimated_min_vram_gb: The candidate's VRAM estimate in
            gigabytes, produced by
            :func:`math_lora.model_selector.estimate_min_vram_gb`. May be
            ``None`` when the caller (task 2.4) chooses not to estimate
            for an ineligible candidate; the function tolerates either
            choice and produces a coherent entry.
        available_vram_gb: Available VRAM per GPU, typically taken from
            :attr:`HardwareProfile.vram_per_gpu_gb`.
        weights: Optional override for the scoring weights.

    Returns:
        A :class:`CandidateEntry` whose fields satisfy Property 2 (a)-(f)
        for the single candidate it describes.

    Determinism:
        Pure function. Identical inputs -> identical output dataclass.
    """

    # ---- eligibility ------------------------------------------------------
    eligibility = check_eligibility(payload)

    # ``model_id`` must be reportable even when ineligible: otherwise we
    # could not key the report row by it. Fall back to the empty string
    # when truly absent so the field is total over all inputs.
    model_id_raw = payload.get("model_id")
    model_id: str = str(model_id_raw) if model_id_raw is not None else ""

    # Optional fields read defensively so the function never raises just
    # because a payload is incomplete.
    declared_param_count_b = (
        float(payload["param_count_b"])
        if "param_count_b" in payload and payload["param_count_b"] is not None
        else None
    )
    license_id = payload.get("license_id")
    license_id_str: str | None = str(license_id) if license_id is not None else None
    license_allows_ft = bool(payload.get("license_allows_finetuning", False))
    license_allows_redist = bool(
        payload.get("license_allows_adapter_redistribution", False)
    )
    license_allows_commercial = bool(payload.get("license_allows_commercial_use", False))

    if not eligibility.eligible:
        # Ineligible candidates are excluded from ranking -- they have no
        # score. Per Requirement 1.10, we still include them in the report
        # so the operator knows why each candidate was excluded.
        return CandidateEntry(
            model_id=model_id,
            declared_param_count_b=declared_param_count_b,
            estimated_min_vram_gb=(
                float(estimated_min_vram_gb)
                if estimated_min_vram_gb is not None
                else None
            ),
            feasible=False,
            vram_shortfall_gb=None,
            score=None,
            license_allows_finetuning=license_allows_ft,
            license_allows_adapter_redistribution=license_allows_redist,
            license_allows_commercial_use=license_allows_commercial,
            license_id=license_id_str,
            eligible=False,
            missing_fields=eligibility.missing_fields,
        )

    # ---- feasibility ------------------------------------------------------
    if estimated_min_vram_gb is None:
        # An eligible candidate must always have a VRAM estimate from the
        # orchestrator -- the only way ``estimated_min_vram_gb is None``
        # is a programming error in the caller, so we surface it loudly.
        raise ValueError(
            "evaluate_candidate: estimated_min_vram_gb is required for "
            f"eligible candidate {model_id!r}"
        )
    feasibility = check_feasibility(
        estimated_min_vram_gb=float(estimated_min_vram_gb),
        available_vram_gb=float(available_vram_gb),
    )

    # ---- scoring (only when feasible) -------------------------------------
    if not feasibility.feasible:
        # Infeasible candidates are excluded from ranking per the
        # design's *Components / 1. Model_Selector* note: "rank feasible
        # candidates ... candidates with ``score is None`` are not part
        # of the ranking". Property 2(c) pins ``vram_shortfall_gb`` to
        # the rounded gap.
        return CandidateEntry(
            model_id=model_id,
            declared_param_count_b=declared_param_count_b,
            estimated_min_vram_gb=float(estimated_min_vram_gb),
            feasible=False,
            vram_shortfall_gb=feasibility.vram_shortfall_gb,
            score=None,
            license_allows_finetuning=license_allows_ft,
            license_allows_adapter_redistribution=license_allows_redist,
            license_allows_commercial_use=license_allows_commercial,
            license_id=license_id_str,
            eligible=True,
            missing_fields=(),
        )

    # Eligible AND feasible: score the candidate.
    permissiveness = license_permissiveness(
        allows_finetuning=license_allows_ft,
        allows_adapter_redistribution=license_allows_redist,
        allows_commercial_use=license_allows_commercial,
    )
    score_value = compute_score(
        ScoringInputs(
            baseline_gsm8k=float(payload["baseline_gsm8k"]),
            baseline_math=float(payload["baseline_math"]),
            param_count_b=float(payload["param_count_b"]),
            license_permissiveness=permissiveness,
        ),
        weights=weights,
    )

    return CandidateEntry(
        model_id=model_id,
        declared_param_count_b=declared_param_count_b,
        estimated_min_vram_gb=float(estimated_min_vram_gb),
        feasible=True,
        vram_shortfall_gb=None,
        score=score_value,
        license_allows_finetuning=license_allows_ft,
        license_allows_adapter_redistribution=license_allows_redist,
        license_allows_commercial_use=license_allows_commercial,
        license_id=license_id_str,
        eligible=True,
        missing_fields=(),
    )


# ---------------------------------------------------------------------------
# Tie-breaker rule (recorded in the selection report by task 2.4)
# ---------------------------------------------------------------------------

#: Human-readable description of the tie-breaker rule, recorded verbatim in
#: the ``SelectionReport.tie_breaker_rule`` field by the task 2.4
#: orchestrator. Pinned here as a module-level constant so the
#: implementation and the report stay in sync.
TIE_BREAKER_RULE: Final[str] = (
    "When two or more candidates produce identical scores, the candidate with "
    "the lexicographically smallest model_id wins."
)


__all__ = [
    "REQUIRED_FIELDS",
    "TIE_BREAKER_RULE",
    "EligibilityResult",
    "FeasibilityResult",
    "ScoringWeights",
    "ScoringInputs",
    "ScoredCandidate",
    "CandidateEntry",
    "check_eligibility",
    "check_feasibility",
    "compute_score",
    "license_permissiveness",
    "pick_winner",
    "evaluate_candidate",
]
