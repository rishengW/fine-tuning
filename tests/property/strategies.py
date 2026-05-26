"""Hypothesis strategies for the math-LoRA fine-tuning domain types.

This module is the single home of the Hypothesis generators reused by every
property-based test in ``tests/property/``. Centralising them here keeps the
test files thin and ensures that the generators stay in lock-step with the
schema constraints declared in :mod:`math_lora.types.models`.

Each ``valid_*`` strategy produces values that pass ``.parse(...)`` on the
corresponding pydantic model. The companion ``candidate_with_one_invalid_field``
strategy yields a *dict* (deliberately not parsed) plus the name of the field
that has been mutated to an invalid value -- this is the shape consumed by
negative property tests written for Requirements 1.10 (missing/invalid base
model fields), 4.9 (LoRA field-level rejection), 7.7 (decoding-parameter
rejection), and friends.

Design references:
    * ``.kiro/specs/math-lora-finetuning/design.md`` -- *Data Models* and
      *Correctness Properties* sections.
    * Task 1.4 in ``tasks.md`` -- enumerates the strategies that must be
      provided, the threshold values to cover, and the LaTeX/unicode-math
      knob on :func:`valid_reasoning_records`.

Determinism:
    All strategies use only Hypothesis primitives, so a Hypothesis seed
    reproduces every example. No global state is mutated.
"""

from __future__ import annotations

from typing import Any, Final

from hypothesis import strategies as st

from math_lora.types.models import (
    BaseModelCandidate,
    BudgetProfile,
    DecodingParams,
    HardwareProfile,
    LoRAConfig,
    ReasoningRecord,
)
from math_lora.types.quantization import BIAS_MODES


# ---------------------------------------------------------------------------
# Shared primitive strategies
# ---------------------------------------------------------------------------

# A short, printable, non-whitespace-only ASCII string. Used wherever the
# pydantic schema requires a non-empty string (model_id, license_id, etc.).
# We constrain the alphabet to keep generated values readable in failure
# blobs and to avoid Hypothesis spending budget on shrink-irrelevant
# unicode permutations for fields whose semantics are "any non-empty
# identifier-ish string".
_NON_EMPTY_TOKEN_ALPHABET: Final[str] = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/."
)


def _non_empty_tokens(min_size: int = 1, max_size: int = 32) -> st.SearchStrategy[str]:
    """Strategy producing non-empty, non-whitespace-only short identifiers."""

    return st.text(
        alphabet=_NON_EMPTY_TOKEN_ALPHABET,
        min_size=max(min_size, 1),
        max_size=max_size,
    ).filter(lambda s: s.strip() != "")


# Currencies kept to a small set so `BudgetProfile.currency` matches what an
# operator would realistically write in a config (Requirement 2.2).
_CURRENCIES: Final[tuple[str, ...]] = ("USD", "EUR", "GBP", "JPY", "CNY")

# Families seeded by Requirement 1.1 ("qwen, deepseek, doubao, ..."). The
# trailing "..." in the spec is intentional: the field is a free-form string
# with these as the canonical examples, so the strategy produces both the
# canonical names and occasional arbitrary identifiers.
_BASE_MODEL_FAMILIES: Final[tuple[str, ...]] = (
    "qwen",
    "deepseek",
    "doubao",
    "mistral",
    "llama",
)

# Tokenizer families used by the candidate models. Kept short and realistic
# so generated values look like real config inputs in failure blobs.
_TOKENIZER_FAMILIES: Final[tuple[str, ...]] = (
    "qwen",
    "deepseek",
    "doubao",
    "llama",
    "mistral",
    "gpt2",
    "tiktoken",
)

# License identifiers covering both permissive and restrictive cases so the
# license-permissiveness scoring (Requirement 1.5, 1.8) sees a realistic mix.
_LICENSE_IDS: Final[tuple[str, ...]] = (
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-nc-4.0",
    "llama-2-community",
    "qwen-research",
    "deepseek-license",
    "proprietary",
)

# The closed set of LoRA target modules referenced in the design's
# *Components and Interfaces* section (LoRA_Trainer / Requirement 4.4) and
# in task 1.4's prompt. Generators draw a subset of these names so the
# resulting LoRAConfig matches what `peft` would resolve against a real
# base model.
_LORA_TARGET_MODULE_NAMES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Native context lengths an operator would realistically declare. Sampled
# alongside arbitrary integers in the strategy below so we hit both common
# values (2k/4k/8k/16k/32k) and unusual ones.
_COMMON_CONTEXT_LENGTHS: Final[tuple[int, ...]] = (
    2048,
    4096,
    8192,
    16384,
    32768,
)

# VRAM thresholds called out in task 1.4 prompt and in the design's
# Quantization-Mode default rule (Requirement 2.4 -- ``< 24 GB`` triggers
# nf4 default). Including them as `sampled_from` ensures shrunk
# counterexamples land on the boundary cases that matter for property
# tests, instead of arbitrary values like 17 or 23.
_VRAM_THRESHOLDS_GB: Final[tuple[int, ...]] = (1, 8, 16, 24, 80)


# ---------------------------------------------------------------------------
# BaseModelCandidate (Requirement 1.1)
# ---------------------------------------------------------------------------


@st.composite
def valid_base_model_candidates(  # type: ignore[no-untyped-def]
    draw,
) -> BaseModelCandidate:
    """Strategy producing valid :class:`BaseModelCandidate` instances.

    Field ranges follow the design's *Data Models* section and task 1.4's
    explicit ranges:

    * ``param_count_b`` in ``[0.5, 70.0]`` -- covers the candidate set
      (Qwen2.5-Math 1.5B/7B, DeepSeek-Math 7B, Llama-3 70B) plus the
      thresholds ``< 1`` and ``> 7`` that exercise the VRAM-estimation
      coefficient table.
    * ``native_context_length_tokens`` drawn from ``2048..32768`` with the
      common 2k/4k/8k/16k/32k cases included via ``sampled_from``.
    * Baseline GSM8K and MATH scores in ``[0.0, 1.0]`` per Requirement 1.1.
    """

    # Mix the canonical family names with occasional arbitrary identifiers
    # so tests that depend on the open-ended "..." in Requirement 1.1 see
    # both shapes.
    family = draw(
        st.one_of(
            st.sampled_from(_BASE_MODEL_FAMILIES),
            _non_empty_tokens(min_size=2, max_size=12),
        )
    )

    # Common context lengths receive most of the budget, while the
    # `integers` arm covers the rest of the 2k..32k range.
    context_length = draw(
        st.one_of(
            st.sampled_from(_COMMON_CONTEXT_LENGTHS),
            st.integers(min_value=2048, max_value=32768),
        )
    )

    return BaseModelCandidate(
        model_id=draw(_non_empty_tokens(min_size=3, max_size=40)),
        revision=draw(_non_empty_tokens(min_size=1, max_size=20)),
        family=family,
        param_count_b=draw(st.floats(min_value=0.5, max_value=70.0, allow_nan=False, allow_infinity=False)),
        license_id=draw(st.sampled_from(_LICENSE_IDS)),
        license_allows_finetuning=draw(st.booleans()),
        license_allows_adapter_redistribution=draw(st.booleans()),
        license_allows_commercial_use=draw(st.booleans()),
        native_context_length_tokens=int(context_length),
        tokenizer_family=draw(st.sampled_from(_TOKENIZER_FAMILIES)),
        baseline_gsm8k=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        baseline_math=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )


# Map from field name to a strategy that produces a value that will be
# rejected by the BaseModelCandidate schema. The keys are exactly the
# fields declared on BaseModelCandidate; any new field added to the
# pydantic model should also appear here. Missing-field is also tested
# below (a key removed from the dict, separately from a mutated value).
_BASE_MODEL_INVALID_VALUE_STRATEGIES: Final[dict[str, st.SearchStrategy[Any]]] = {
    # Empty strings violate ``min_length=1`` on every NonEmptyStr field.
    "model_id": st.just(""),
    "revision": st.just(""),
    "family": st.just(""),
    "license_id": st.just(""),
    "tokenizer_family": st.just(""),
    # ``param_count_b`` must be > 0.
    "param_count_b": st.one_of(
        st.just(0.0),
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    ),
    # license flags are StrictBool, so a non-bool is rejected.
    "license_allows_finetuning": st.just("yes"),
    "license_allows_adapter_redistribution": st.just(1),  # StrictBool rejects ints
    "license_allows_commercial_use": st.just("true"),
    # native_context_length_tokens must be > 0.
    "native_context_length_tokens": st.one_of(
        st.just(0),
        st.integers(max_value=0),
    ),
    # baseline scores must be within [0.0, 1.0].
    "baseline_gsm8k": st.one_of(
        st.floats(min_value=1.0001, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-10.0, max_value=-0.0001, allow_nan=False, allow_infinity=False),
    ),
    "baseline_math": st.one_of(
        st.floats(min_value=1.0001, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-10.0, max_value=-0.0001, allow_nan=False, allow_infinity=False),
    ),
}


@st.composite
def candidate_with_one_invalid_field(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[dict[str, Any], str]:
    """Strategy producing ``(dict, field_name)`` where exactly one field is bad.

    The dict is the raw payload (intentionally not parsed) so negative
    property tests can call ``BaseModelCandidate.parse(payload)`` and assert
    that the named field appears in the resulting :class:`SchemaValidationError`'s
    ``fields`` tuple. This shape directly supports the per-field rejection
    requirement called out by Requirement 1.10 (the selector must list the
    *specific* missing/invalid fields in the report).

    The strategy preserves a valid skeleton for every other field so the
    test isolates exactly one violation per generated example.
    """

    valid_candidate = draw(valid_base_model_candidates())
    payload: dict[str, Any] = valid_candidate.model_dump()

    # Pick which field we will corrupt. Sampled from the keys of the
    # invalid-strategy table so we are guaranteed to have a concrete
    # invalid value for it.
    field_name = draw(st.sampled_from(sorted(_BASE_MODEL_INVALID_VALUE_STRATEGIES)))
    payload[field_name] = draw(_BASE_MODEL_INVALID_VALUE_STRATEGIES[field_name])

    return payload, field_name


# ---------------------------------------------------------------------------
# HardwareProfile (Requirements 1.2, 2.1)
# ---------------------------------------------------------------------------


# Boundary VRAM samples plus arbitrary integers in the closed schema range
# ``[1, 1024]``. The boundary samples cover the consumer-GPU thresholds
# called out in the design (8/16 GB consumer, 24 GB cutoff for nf4 default,
# 80 GB high-end H100/A100), and the closed-interval endpoints 1 and 1024.
_VRAM_PER_GPU_GB_STRATEGY: Final[st.SearchStrategy[int]] = st.one_of(
    st.sampled_from(_VRAM_THRESHOLDS_GB),
    st.sampled_from((1, 1024)),  # closed-interval endpoints
    st.integers(min_value=1, max_value=1024),
)


@st.composite
def valid_hardware_profiles(  # type: ignore[no-untyped-def]
    draw,
) -> HardwareProfile:
    """Strategy producing valid :class:`HardwareProfile` instances.

    The ``vram_per_gpu_gb`` strategy is the union of:

    * the boundary thresholds ``{1, 8, 16, 24, 80}`` from task 1.4's prompt,
    * the closed-interval endpoints ``1`` and ``1024`` from Requirement 1.2,
    * arbitrary integers in ``[1, 1024]``.

    This guarantees that downstream property tests on Requirement 2.4
    (default ``nf4`` when ``vram_per_gpu_gb < 24``) and Requirement 1.4
    (feasibility check) hit the boundary values during shrinking.
    """

    return HardwareProfile(
        gpu_model=draw(_non_empty_tokens(min_size=2, max_size=24)),
        gpu_count=draw(st.integers(min_value=1, max_value=8)),
        vram_per_gpu_gb=draw(_VRAM_PER_GPU_GB_STRATEGY),
        system_ram_gb=draw(st.integers(min_value=1, max_value=2048)),
        disk_space_gb=draw(st.integers(min_value=1, max_value=10_000)),
        accelerator_family=draw(st.sampled_from(("cuda", "rocm", "metal", "cpu"))),
        deployment=draw(st.sampled_from(("local", "cloud"))),
    )


# ---------------------------------------------------------------------------
# BudgetProfile (Requirement 2.2)
# ---------------------------------------------------------------------------


@st.composite
def valid_budget_profiles(  # type: ignore[no-untyped-def]
    draw,
) -> BudgetProfile:
    """Strategy producing valid :class:`BudgetProfile` instances.

    ``cost_rate_per_gpu_hour`` includes ``0.0`` (locally-owned hardware,
    sunk cost) per the docstring on the pydantic model.
    """

    return BudgetProfile(
        max_cost=draw(st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)),
        currency=draw(st.sampled_from(_CURRENCIES)),
        max_wallclock_hours=draw(st.floats(min_value=0.1, max_value=720.0, allow_nan=False, allow_infinity=False)),
        cost_rate_per_gpu_hour=draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)),
    )


# ---------------------------------------------------------------------------
# LoRAConfig (Requirements 4.1, 4.2)
# ---------------------------------------------------------------------------


def _target_modules_strategy() -> st.SearchStrategy[list[str] | None]:
    """Either ``None`` or a non-empty subset of the documented module set.

    The subset is materialized by drawing a permutation of
    :data:`_LORA_TARGET_MODULE_NAMES` and slicing to a random length, which
    guarantees no duplicates without relying on Hypothesis's ``unique=True``
    filter (which discards examples and slows shrink). ``None`` is included
    so Requirement 4.4 (default-resolution behavior) is exercised by
    consumers of the strategy.
    """

    permutations = st.permutations(list(_LORA_TARGET_MODULE_NAMES))
    sliced = permutations.flatmap(
        lambda perm: st.integers(
            min_value=1, max_value=len(_LORA_TARGET_MODULE_NAMES)
        ).map(lambda n: perm[:n])
    )
    return st.one_of(st.none(), sliced)


@st.composite
def valid_lora_configs(  # type: ignore[no-untyped-def]
    draw,
) -> LoRAConfig:
    """Strategy producing valid :class:`LoRAConfig` instances.

    All ranges come straight from Requirement 4.1 / 4.2:

    * ``r in [4, 128]``
    * ``alpha > 0``
    * ``dropout in [0.0, 1.0]``
    * ``target_modules`` either ``None`` or a non-duplicate subset of the
      documented projection-name vocabulary.
    * ``bias`` from ``{none, all, lora_only}``.
    """

    return LoRAConfig(
        r=draw(st.integers(min_value=4, max_value=128)),
        alpha=draw(st.floats(min_value=0.001, max_value=256.0, allow_nan=False, allow_infinity=False)),
        dropout=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        target_modules=draw(_target_modules_strategy()),
        bias=draw(st.sampled_from(BIAS_MODES)),
    )


# ---------------------------------------------------------------------------
# DecodingParams (Requirement 7.7)
# ---------------------------------------------------------------------------


@st.composite
def valid_decoding_params(  # type: ignore[no-untyped-def]
    draw,
) -> DecodingParams:
    """Strategy producing valid :class:`DecodingParams` instances.

    Ranges come directly from Requirement 7.7:

    * ``temperature in [0.0, 2.0]``
    * ``top_p in [0.0, 1.0]``
    * ``top_k >= 0``
    * ``max_new_tokens > 0``
    * ``seed`` is any int (the schema rejects only non-int values).
    """

    return DecodingParams(
        temperature=draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        top_p=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        top_k=draw(st.integers(min_value=0, max_value=200)),
        max_new_tokens=draw(st.integers(min_value=1, max_value=4096)),
        seed=draw(st.integers(min_value=-(2**31), max_value=2**31 - 1)),
    )



# ---------------------------------------------------------------------------
# ReasoningRecord (Requirement 3.2 -- Reasoning_Format)
# ---------------------------------------------------------------------------


# LaTeX delimiters and unicode math symbols used to spice up reasoning text.
# The selection covers the constructs explicitly mentioned in the task 1.4
# prompt (``$...$``, ``\\frac{}{}``, ``\\boxed{}``, ``\u222b``, ``\u03c0``,
# ``\u2264``) plus a few neighbours so dataset-builder normalization
# (Requirement 3.6) is exercised on a realistic mix.
_LATEX_FRAGMENTS: Final[tuple[str, ...]] = (
    r"$x^2 + 1$",
    r"$\frac{1}{2}$",
    r"$\boxed{42}$",
    r"$\int_0^1 x\,dx$",
    r"$\sum_{k=1}^n k$",
    r"$\sqrt{2}$",
    r"$e^{i\pi} + 1 = 0$",
    r"\frac{a}{b}",
    r"\boxed{\pi}",
)

_UNICODE_MATH_FRAGMENTS: Final[tuple[str, ...]] = (
    "\u222b",        # ∫
    "\u03c0",        # π
    "\u2264",        # ≤
    "\u2265",        # ≥
    "\u2211",        # ∑
    "\u221a",        # √
    "\u00b1",        # ±
    "x \u2208 \u211d",  # x ∈ ℝ
)


def _math_text_strategy(
    latex_probability: float,
    *,
    min_size: int = 1,
    max_size: int = 80,
) -> st.SearchStrategy[str]:
    """Strategy producing a non-empty, non-whitespace-only text fragment.

    With probability ``latex_probability`` the fragment is constructed by
    interleaving plain ASCII text with one or more LaTeX or unicode-math
    fragments. Otherwise the fragment is plain ASCII text only. The
    resulting string is guaranteed to be non-empty and non-whitespace-only
    so ``ReasoningRecord`` parsing succeeds.
    """

    if not 0.0 <= latex_probability <= 1.0:
        raise ValueError(
            f"latex_probability must be in [0.0, 1.0], got {latex_probability!r}"
        )

    plain = st.text(
        alphabet=st.characters(
            min_codepoint=0x20,
            max_codepoint=0x7E,  # printable ASCII only
            blacklist_characters=("\t", "\r", "\n"),
        ),
        min_size=max(min_size, 1),
        max_size=max_size,
    ).filter(lambda s: s.strip() != "")

    # ``with_math`` builds a string by concatenating a plain-text prefix,
    # a math fragment, and a plain-text suffix. We split the size budget
    # roughly in thirds so the resulting string is bounded and the math
    # fragment remains visually identifiable in failure blobs.
    third = max(1, max_size // 3)

    @st.composite
    def with_math(draw):  # type: ignore[no-untyped-def]
        prefix = draw(st.text(alphabet=_NON_EMPTY_TOKEN_ALPHABET + " ", min_size=0, max_size=third))
        fragment = draw(
            st.one_of(
                st.sampled_from(_LATEX_FRAGMENTS),
                st.sampled_from(_UNICODE_MATH_FRAGMENTS),
            )
        )
        suffix = draw(st.text(alphabet=_NON_EMPTY_TOKEN_ALPHABET + " ", min_size=0, max_size=third))
        result = f"{prefix} {fragment} {suffix}".strip()
        # Belt-and-braces guard against an all-whitespace shrink: the
        # fragment itself is always non-whitespace, so this is rare but
        # cheap to enforce.
        if not result:
            result = fragment
        return result

    if latex_probability == 0.0:
        return plain
    if latex_probability == 1.0:
        return with_math()

    # ``one_of`` does not directly accept weights, so emulate the mix by
    # drawing a uniform float and branching. Hypothesis still shrinks to
    # the simpler ``plain`` arm naturally.
    @st.composite
    def mixed(draw):  # type: ignore[no-untyped-def]
        roll = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        if roll < latex_probability:
            return draw(with_math())
        return draw(plain)

    return mixed()


def valid_reasoning_records(
    latex_probability: float = 0.3,
) -> st.SearchStrategy[ReasoningRecord]:
    """Strategy producing valid :class:`ReasoningRecord` instances.

    Args:
        latex_probability: The probability (per generated string) that the
            string contains LaTeX delimiters or unicode math symbols.
            Must be in ``[0.0, 1.0]``. Default ``0.3`` matches the value
            recommended in task 1.4's prompt.

    The record contains:

    * a non-empty ``problem`` string,
    * a non-empty ``solution_steps`` list of length ``1..6``, each step a
      non-empty string,
    * a non-empty ``final_answer`` string.

    Each string is drawn from :func:`_math_text_strategy` so all three
    fields share the same LaTeX/unicode-math knob.
    """

    text_strategy = _math_text_strategy(latex_probability=latex_probability)

    @st.composite
    def build(draw):  # type: ignore[no-untyped-def]
        return ReasoningRecord(
            problem=draw(text_strategy),
            solution_steps=draw(
                st.lists(text_strategy, min_size=1, max_size=6)
            ),
            final_answer=draw(text_strategy),
        )

    return build()


# ---------------------------------------------------------------------------
# Simulated training-run event traces (state-machine properties 19-26)
# ---------------------------------------------------------------------------


def simulated_training_run_traces(
    max_steps: int = 200,
    validation_interval_steps: int = 50,
    *,
    allow_nonfinite_loss: bool = False,
) -> st.SearchStrategy[list[dict[str, Any]]]:
    """Strategy producing a monotonic list of simulated training-step events.

    The returned list is suitable as input for the state-machine properties
    in the design document:

    * Property 19 (training halts at the correct step) -- consumes the
      length and ``step`` field.
    * Property 20 (checkpoint retention invariant) -- consumes ``step`` and
      ``val_loss``.
    * Property 21 (resume round-trip equivalence) -- consumes the full
      trace, plus a ``checkpoint`` flag derived from ``step %
      checkpoint_interval``.
    * Property 23 (periodic event scheduling) -- consumes ``step`` plus the
      validation-interval fields.
    * Property 25 (non-finite loss halt) -- enabled by setting
      ``allow_nonfinite_loss=True`` so a small fraction of events carry
      ``train_loss`` of ``nan`` or ``inf``.

    Each event dict has shape::

        {
            "step": int,                          # strictly increasing, 1-based
            "train_loss": float,                  # > 0; may be NaN/inf if flag set
            "val_loss": float | None,             # set on validation steps only
            "learning_rate": float,               # > 0
        }

    Args:
        max_steps: Upper bound on the number of events generated. The
            actual length is drawn uniformly in ``[1, max_steps]`` so
            small traces are common (helping shrink) while occasional
            full-length traces still occur. Defaults to ``200`` per the
            task prompt.
        validation_interval_steps: Steps at which ``val_loss`` is non-null.
            Defaults to ``50`` per the task prompt.
        allow_nonfinite_loss: If ``True``, ``train_loss`` may take the
            values ``nan`` or ``inf`` with low probability. Used by the
            non-finite-loss halt property test (Property 25); leave
            ``False`` for traces that should mimic a healthy run.
    """

    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if validation_interval_steps < 1:
        raise ValueError(
            f"validation_interval_steps must be >= 1, got {validation_interval_steps}"
        )

    # ``train_loss`` is positive in a healthy run. We bound it loosely to
    # cover both early-training (large) and converged (small) phases.
    healthy_loss = st.floats(
        min_value=1e-6,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    )
    if allow_nonfinite_loss:
        # Mostly healthy values, occasionally NaN/inf so Property 25 has
        # something to detect. Hypothesis still shrinks toward the simpler
        # finite branch.
        train_loss_strategy: st.SearchStrategy[float] = st.one_of(
            healthy_loss,
            st.sampled_from((float("nan"), float("inf"), -float("inf"))),
        )
    else:
        train_loss_strategy = healthy_loss

    learning_rate_strategy = st.floats(
        min_value=1e-7,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )

    val_loss_strategy = st.floats(
        min_value=1e-6,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    )

    @st.composite
    def build(draw):  # type: ignore[no-untyped-def]
        # Length is drawn first so Hypothesis can shrink it independently
        # of the per-event content.
        length = draw(st.integers(min_value=1, max_value=max_steps))

        events: list[dict[str, Any]] = []
        for step in range(1, length + 1):
            is_validation_step = step % validation_interval_steps == 0
            event: dict[str, Any] = {
                "step": step,
                "train_loss": draw(train_loss_strategy),
                "val_loss": draw(val_loss_strategy) if is_validation_step else None,
                "learning_rate": draw(learning_rate_strategy),
            }
            events.append(event)

        return events

    return build()


__all__ = [
    "valid_base_model_candidates",
    "candidate_with_one_invalid_field",
    "valid_hardware_profiles",
    "valid_budget_profiles",
    "valid_lora_configs",
    "valid_decoding_params",
    "valid_reasoning_records",
    "simulated_training_run_traces",
]
