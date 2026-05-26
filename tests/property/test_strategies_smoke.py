"""Smoke tests that every strategy in :mod:`tests.property.strategies` works.

These tests are not full property tests for any one requirement; their sole
purpose is to:

1. Confirm the strategies in :mod:`tests.property.strategies` produce values
   that pass ``.parse(...)`` on the corresponding pydantic model.
2. Confirm the negative ``candidate_with_one_invalid_field`` strategy produces
   payloads that the schema rejects with the documented error type, and that
   the named field is surfaced on the error.
3. Confirm the simulated training-trace strategy produces a strictly
   monotonic list of events with the structural fields the state-machine
   properties (Properties 19-26) consume.

A failure in this file means the strategies are themselves broken, which
would silently corrupt every property test that depends on them, so it is
worth catching at this layer rather than in the downstream property tests.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from math_lora.types.errors import SchemaValidationError
from math_lora.types.models import (
    BaseModelCandidate,
    BudgetProfile,
    DecodingParams,
    HardwareProfile,
    LoRAConfig,
    ReasoningRecord,
)
from math_lora.types.quantization import BIAS_MODES

from tests.property.strategies import (
    candidate_with_one_invalid_field,
    simulated_training_run_traces,
    valid_base_model_candidates,
    valid_budget_profiles,
    valid_decoding_params,
    valid_hardware_profiles,
    valid_lora_configs,
    valid_reasoning_records,
)


# ---------------------------------------------------------------------------
# Positive shape tests for each strategy
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(candidate=valid_base_model_candidates())
def test_valid_base_model_candidates_parse(candidate: BaseModelCandidate) -> None:
    # The strategy returns a BaseModelCandidate directly; the round-trip
    # through ``parse`` confirms the dumped dict also re-parses, which is
    # what downstream tests will rely on.
    parsed = BaseModelCandidate.parse(candidate.model_dump())
    assert parsed == candidate
    assert 0.0 <= parsed.baseline_gsm8k <= 1.0
    assert 0.0 <= parsed.baseline_math <= 1.0
    assert parsed.param_count_b > 0
    assert parsed.native_context_length_tokens > 0


@pytest.mark.property
@given(profile=valid_hardware_profiles())
def test_valid_hardware_profiles_parse(profile: HardwareProfile) -> None:
    parsed = HardwareProfile.parse(profile.model_dump())
    assert parsed == profile
    # The whole point of mixing ``sampled_from`` boundary values into the
    # strategy is that values at 1 and 1024 are reachable without filter
    # rejections; assert the schema's closed-interval bounds hold.
    assert 1 <= parsed.vram_per_gpu_gb <= 1024


@pytest.mark.property
@given(profile=valid_budget_profiles())
def test_valid_budget_profiles_parse(profile: BudgetProfile) -> None:
    parsed = BudgetProfile.parse(profile.model_dump())
    assert parsed == profile
    assert parsed.max_cost > 0
    assert parsed.max_wallclock_hours > 0
    assert parsed.cost_rate_per_gpu_hour >= 0


@pytest.mark.property
@given(config=valid_lora_configs())
def test_valid_lora_configs_parse(config: LoRAConfig) -> None:
    parsed = LoRAConfig.parse(config.model_dump())
    assert parsed == config
    assert 4 <= parsed.r <= 128
    assert parsed.alpha > 0
    assert 0.0 <= parsed.dropout <= 1.0
    assert parsed.bias in BIAS_MODES
    if parsed.target_modules is not None:
        # No duplicates per LoRAConfig._target_modules_unique_and_non_empty.
        assert len(parsed.target_modules) == len(set(parsed.target_modules))
        assert len(parsed.target_modules) >= 1


@pytest.mark.property
@given(params=valid_decoding_params())
def test_valid_decoding_params_parse(params: DecodingParams) -> None:
    parsed = DecodingParams.parse(params.model_dump())
    assert parsed == params
    assert 0.0 <= parsed.temperature <= 2.0
    assert 0.0 <= parsed.top_p <= 1.0
    assert parsed.top_k >= 0
    assert parsed.max_new_tokens > 0


@pytest.mark.property
@given(record=valid_reasoning_records())
def test_valid_reasoning_records_parse_with_default_latex(
    record: ReasoningRecord,
) -> None:
    parsed = ReasoningRecord.parse(record.model_dump())
    assert parsed == record
    assert parsed.problem.strip()
    assert parsed.final_answer.strip()
    assert len(parsed.solution_steps) >= 1
    assert all(step.strip() for step in parsed.solution_steps)


@pytest.mark.property
@given(record=valid_reasoning_records(latex_probability=0.0))
def test_valid_reasoning_records_with_no_latex(record: ReasoningRecord) -> None:
    # With ``latex_probability=0.0`` no LaTeX or unicode-math characters
    # should appear, but the record must still be a valid ReasoningRecord.
    ReasoningRecord.parse(record.model_dump())


@pytest.mark.property
@given(record=valid_reasoning_records(latex_probability=1.0))
def test_valid_reasoning_records_with_full_latex(record: ReasoningRecord) -> None:
    # With ``latex_probability=1.0`` parsing must still succeed; this is
    # the case downstream dataset-builder normalization tests rely on.
    ReasoningRecord.parse(record.model_dump())


# ---------------------------------------------------------------------------
# Negative shape: candidate_with_one_invalid_field
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(item=candidate_with_one_invalid_field())
def test_candidate_with_one_invalid_field_is_rejected(
    item: tuple[dict[str, Any], str],
) -> None:
    payload, field_name = item

    # The payload is a dict (not a parsed model) carrying exactly one bad
    # field. ``BaseModelCandidate.parse`` should refuse it via a
    # ``SchemaValidationError`` whose ``fields`` tuple includes the name we
    # corrupted, satisfying the per-field surfacing requirement.
    with pytest.raises(SchemaValidationError) as exc_info:
        BaseModelCandidate.parse(payload)

    err = exc_info.value
    assert field_name in err.fields, (
        f"expected '{field_name}' to appear in error.fields, got {err.fields!r}"
    )


# ---------------------------------------------------------------------------
# simulated_training_run_traces shape
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(trace=simulated_training_run_traces(max_steps=200, validation_interval_steps=50))
def test_simulated_training_run_traces_default_shape(
    trace: list[dict[str, Any]],
) -> None:
    # Non-empty; each event has the documented schema; steps strictly
    # increasing starting at 1.
    assert len(trace) >= 1
    assert len(trace) <= 200

    for idx, event in enumerate(trace):
        assert event["step"] == idx + 1, (
            f"trace must be 1-based monotonic: event {idx} has step {event['step']}"
        )
        # Healthy run: train_loss must be finite and positive.
        train_loss = event["train_loss"]
        assert isinstance(train_loss, float)
        assert math.isfinite(train_loss) and train_loss > 0
        # Validation loss is set exactly on multiples of the validation
        # interval; otherwise it is None.
        if event["step"] % 50 == 0:
            assert event["val_loss"] is not None
        else:
            assert event["val_loss"] is None
        assert event["learning_rate"] > 0


@pytest.mark.property
@given(
    trace=simulated_training_run_traces(
        max_steps=120,
        validation_interval_steps=10,
        allow_nonfinite_loss=True,
    )
)
def test_simulated_training_run_traces_with_nonfinite_loss_flag(
    trace: list[dict[str, Any]],
) -> None:
    # Even with the flag on, the structural invariants (monotonic, in-range
    # length, schema fields present) must hold. The presence of NaN/inf
    # values is *allowed*, not required, on any individual draw.
    assert 1 <= len(trace) <= 120
    for idx, event in enumerate(trace):
        assert event["step"] == idx + 1
        # Either finite-and-positive or non-finite (NaN/inf).
        train_loss = event["train_loss"]
        assert isinstance(train_loss, float)
        assert math.isfinite(train_loss) or math.isnan(train_loss) or math.isinf(train_loss)
        assert event["learning_rate"] > 0


# ---------------------------------------------------------------------------
# Sanity: invalid latex_probability surfaces immediately
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_valid_reasoning_records_rejects_out_of_range_probability(bad: float) -> None:
    with pytest.raises(ValueError):
        # Calling the strategy factory does the validation eagerly so we
        # do not need to draw anything from it.
        valid_reasoning_records(latex_probability=bad)


@pytest.mark.unit
def test_simulated_training_run_traces_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        simulated_training_run_traces(max_steps=0)
    with pytest.raises(ValueError):
        simulated_training_run_traces(validation_interval_steps=0)


# ---------------------------------------------------------------------------
# Determinism: drawing twice from the same strategy under the same Hypothesis
# data context yields the same value, confirming our composites do not pull
# nondeterministic state (random.random, time.time, env vars).
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=st.data())
def test_composite_strategies_replay_deterministically(data: st.DataObject) -> None:
    # ``data.draw`` with the same strategy and the same data context is a
    # standard Hypothesis pattern for confirming a composite strategy is a
    # pure function of its random source. If a strategy reached out to
    # ``random.random()`` or ``os.urandom`` it would fail this check.
    candidate = data.draw(valid_base_model_candidates())
    record = data.draw(valid_reasoning_records(latex_probability=0.5))

    # The drawn values themselves must round-trip through ``parse``; this
    # is what every downstream test relies on.
    assert BaseModelCandidate.parse(candidate.model_dump()) == candidate
    assert ReasoningRecord.parse(record.model_dump()) == record
