"""Example-based unit tests for the domain-type schemas defined in task 1.2.

Coverage matrix
---------------

This file covers the five configuration schemas listed by the task prompt:

* :class:`math_lora.types.BaseModelCandidate` (Requirement 1.1)
* :class:`math_lora.types.HardwareProfile`    (Requirements 1.2, 2.1)
* :class:`math_lora.types.BudgetProfile`      (Requirement 2.2)
* :class:`math_lora.types.LoRAConfig`         (Requirements 4.1, 4.2)
* :class:`math_lora.types.DecodingParams`     (Requirement 7.7)

For each schema we exercise:

1. **Positive construction.** A fully-valid payload built through the
   ``.parse()`` classmethod returns the model instance with every field
   populated as supplied.
2. **Boundary acceptance.** The lowest and highest in-range values for each
   numeric field are accepted (e.g. ``vram_per_gpu_gb in [1, 1024]`` -> ``1``
   and ``1024`` are both valid).
3. **Boundary rejection.** Values just outside the closed/half-open
   intervals are rejected (e.g. ``vram_per_gpu_gb=0`` and
   ``vram_per_gpu_gb=1025``). Each rejection asserts both the typed
   error class declared in the design's *Error Handling* table and that the
   offending field name appears in ``.fields``.

The ``.parse()`` classmethod is the public API exposed by every model in
``math_lora.types.models``. ``model_validate`` would only raise
``pydantic.ValidationError``; ``.parse()`` re-raises that as the
component-specific schema error so callers in later tasks (``LoRA_Trainer``,
``Hardware_Budget_Planner``, ``Inference_Server``) do not have to reach into
pydantic internals.
"""

from __future__ import annotations

from typing import Any

import pytest

from math_lora.types import (
    BaseModelCandidate,
    BudgetProfile,
    ConfigLoadError,
    DecodingParams,
    DecodingParamsInvalid,
    HardwareProfile,
    LoRAConfig,
    LoRAConfigInvalid,
    SchemaValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_candidate_payload() -> dict[str, Any]:
    """Return a fully-valid payload for :class:`BaseModelCandidate`.

    Tests mutate one field at a time on a copy of this dict so that every
    rejection test isolates a single field violation -- no rejection should
    pass for the wrong reason because of a co-occurring problem.
    """

    return {
        "model_id": "qwen/Qwen2.5-Math-7B",
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


def _valid_hardware_payload() -> dict[str, Any]:
    return {
        "gpu_model": "RTX 4090",
        "gpu_count": 1,
        "vram_per_gpu_gb": 24,
        "system_ram_gb": 64,
        "disk_space_gb": 1024,
        "accelerator_family": "cuda",
        "deployment": "local",
    }


def _valid_budget_payload() -> dict[str, Any]:
    return {
        "max_cost": 50.0,
        "currency": "USD",
        "max_wallclock_hours": 12.0,
        "cost_rate_per_gpu_hour": 0.5,
    }


def _valid_lora_payload() -> dict[str, Any]:
    return {
        "r": 16,
        "alpha": 32.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }


def _valid_decoding_payload() -> dict[str, Any]:
    return {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 256,
        "seed": 42,
    }


def _with(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a shallow copy of ``payload`` with ``overrides`` applied.

    Used by rejection tests so that the mutated payload differs from the
    valid baseline by exactly one field.
    """

    return {**payload, **overrides}


# ===========================================================================
# BaseModelCandidate (Requirement 1.1)
# ===========================================================================


class TestBaseModelCandidate:
    """Schema tests for :class:`BaseModelCandidate`.

    The type uses the default :class:`SchemaValidationError`; it is the
    input record consumed by ``Model_Selector`` and the *Error Handling*
    table assigns ``EmptyCandidateList`` and ``IneligibleCandidate`` to the
    selector itself rather than to the schema, so a schema rejection here
    surfaces the generic field-level error.
    """

    @pytest.mark.unit
    def test_parse_valid_payload_returns_populated_instance(self) -> None:
        payload = _valid_candidate_payload()

        candidate = BaseModelCandidate.parse(payload)

        # Every field in the payload round-trips.
        assert candidate.model_id == payload["model_id"]
        assert candidate.revision == payload["revision"]
        assert candidate.family == payload["family"]
        assert candidate.param_count_b == payload["param_count_b"]
        assert candidate.license_id == payload["license_id"]
        assert candidate.license_allows_finetuning is True
        assert candidate.license_allows_adapter_redistribution is True
        assert candidate.license_allows_commercial_use is True
        assert candidate.native_context_length_tokens == 4096
        assert candidate.tokenizer_family == "qwen"
        assert candidate.baseline_gsm8k == 0.85
        assert candidate.baseline_math == 0.55

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("baseline_gsm8k", "baseline_math"),
        [
            (0.0, 0.0),  # lower bound of the closed interval
            (1.0, 1.0),  # upper bound of the closed interval
        ],
    )
    def test_baseline_score_boundaries_are_accepted(
        self, baseline_gsm8k: float, baseline_math: float
    ) -> None:
        # Requirement 1.1 declares baseline scores in the closed range
        # [0.0, 1.0]; both endpoints must parse.
        candidate = BaseModelCandidate.parse(
            _with(
                _valid_candidate_payload(),
                baseline_gsm8k=baseline_gsm8k,
                baseline_math=baseline_math,
            )
        )
        assert candidate.baseline_gsm8k == baseline_gsm8k
        assert candidate.baseline_math == baseline_math

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["model_id", "revision", "family", "license_id", "tokenizer_family"])
    def test_non_empty_string_fields_reject_empty_string(self, field: str) -> None:
        # Requirement 1.1 declares every string field non-empty; the
        # NonEmptyStr alias enforces ``min_length=1``.
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(_with(_valid_candidate_payload(), **{field: ""}))
        assert field in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_param_count_b_must_be_strictly_positive(self, bad_value: float) -> None:
        # Requirement 1.1 declares ``param_count_b`` as a positive float
        # (a 0-billion-parameter model is meaningless).
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(
                _with(_valid_candidate_payload(), param_count_b=bad_value)
            )
        assert "param_count_b" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_native_context_length_tokens_must_be_strictly_positive(
        self, bad_value: int
    ) -> None:
        # Requirement 1.1 declares ``native_context_length_tokens > 0``.
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(
                _with(_valid_candidate_payload(), native_context_length_tokens=bad_value)
            )
        assert "native_context_length_tokens" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("baseline_gsm8k", -0.01),
            ("baseline_gsm8k", 1.01),
            ("baseline_math", -0.01),
            ("baseline_math", 1.01),
        ],
    )
    def test_baseline_scores_reject_out_of_range_values(
        self, field: str, bad_value: float
    ) -> None:
        # Requirement 1.1 fixes baseline accuracy in [0.0, 1.0]; values just
        # outside the closed interval must be rejected with a named field.
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(
                _with(_valid_candidate_payload(), **{field: bad_value})
            )
        assert field in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field",
        [
            "license_allows_finetuning",
            "license_allows_adapter_redistribution",
            "license_allows_commercial_use",
        ],
    )
    def test_license_flags_reject_non_boolean_input(self, field: str) -> None:
        # Requirement 1.8 declares three boolean license flags. The fields
        # use ``StrictBool`` so a string ``"yes"`` (a common YAML mistake)
        # is rejected with the field name surfaced.
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(
                _with(_valid_candidate_payload(), **{field: "yes"})
            )
        assert field in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_field",
        [
            "model_id",
            "revision",
            "family",
            "param_count_b",
            "license_id",
            "license_allows_finetuning",
            "license_allows_adapter_redistribution",
            "license_allows_commercial_use",
            "native_context_length_tokens",
            "tokenizer_family",
            "baseline_gsm8k",
            "baseline_math",
        ],
    )
    def test_missing_required_field_is_rejected_with_name(self, missing_field: str) -> None:
        # Requirement 1.10 says the selector must list missing fields. The
        # schema layer enforces presence so the selector receives a
        # populated record, not a half-defaulted one.
        payload = _valid_candidate_payload()
        del payload[missing_field]
        with pytest.raises(SchemaValidationError) as exc_info:
            BaseModelCandidate.parse(payload)
        assert missing_field in exc_info.value.fields


# ===========================================================================
# HardwareProfile (Requirements 1.2, 2.1)
# ===========================================================================


class TestHardwareProfile:
    """Schema tests for :class:`HardwareProfile`.

    The model raises :class:`ConfigLoadError` (Requirement 2.3): a malformed
    or incomplete profile must surface an error that names *both* the file
    and the field. These tests cover the field half of that contract; the
    file-path half is covered by ``Hardware_Budget_Planner`` integration
    tests in task 3.x.
    """

    @pytest.mark.unit
    def test_parse_valid_payload_returns_populated_instance(self) -> None:
        payload = _valid_hardware_payload()

        profile = HardwareProfile.parse(payload)

        assert profile.gpu_model == "RTX 4090"
        assert profile.gpu_count == 1
        assert profile.vram_per_gpu_gb == 24
        assert profile.system_ram_gb == 64
        assert profile.disk_space_gb == 1024
        assert profile.accelerator_family == "cuda"
        assert profile.deployment == "local"

    @pytest.mark.unit
    @pytest.mark.parametrize("vram", [1, 1024])
    def test_vram_per_gpu_gb_accepts_closed_interval_endpoints(self, vram: int) -> None:
        # Requirement 1.2 declares VRAM in the closed range [1, 1024]; both
        # endpoints must parse.
        profile = HardwareProfile.parse(_with(_valid_hardware_payload(), vram_per_gpu_gb=vram))
        assert profile.vram_per_gpu_gb == vram

    @pytest.mark.unit
    @pytest.mark.parametrize("vram", [0, -1, 1025, 2048])
    def test_vram_per_gpu_gb_rejects_out_of_interval_values(self, vram: int) -> None:
        # Values outside [1, 1024] are out-of-range and must surface a
        # named field via ConfigLoadError.
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), vram_per_gpu_gb=vram))
        assert "vram_per_gpu_gb" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("count", [0, -1])
    def test_gpu_count_must_be_at_least_one(self, count: int) -> None:
        # Requirement 2.1 requires at least one GPU per profile.
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), gpu_count=count))
        assert "gpu_count" in exc_info.value.fields

    @pytest.mark.unit
    def test_gpu_count_one_is_accepted(self) -> None:
        # ``gpu_count >= 1`` -- the lower bound is part of the valid range.
        profile = HardwareProfile.parse(_with(_valid_hardware_payload(), gpu_count=1))
        assert profile.gpu_count == 1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("system_ram_gb", 0),
            ("system_ram_gb", -1),
            ("disk_space_gb", 0),
            ("disk_space_gb", -1),
        ],
    )
    def test_ram_and_disk_must_be_strictly_positive(
        self, field: str, bad_value: int
    ) -> None:
        # Requirement 2.1: system RAM and disk are positive integers.
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), **{field: bad_value}))
        assert field in exc_info.value.fields

    @pytest.mark.unit
    def test_gpu_model_rejects_empty_string(self) -> None:
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), gpu_model=""))
        assert "gpu_model" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("family", ["cuda", "rocm", "metal", "cpu"])
    def test_accelerator_family_accepts_each_documented_value(self, family: str) -> None:
        # Design § Data Models / HardwareProfile fixes the allowed set.
        profile = HardwareProfile.parse(_with(_valid_hardware_payload(), accelerator_family=family))
        assert profile.accelerator_family == family

    @pytest.mark.unit
    @pytest.mark.parametrize("family", ["tpu", "CUDA", "", "openxla"])
    def test_accelerator_family_rejects_unknown_values(self, family: str) -> None:
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), accelerator_family=family))
        assert "accelerator_family" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("deployment", ["local", "cloud"])
    def test_deployment_accepts_each_documented_value(self, deployment: str) -> None:
        profile = HardwareProfile.parse(_with(_valid_hardware_payload(), deployment=deployment))
        assert profile.deployment == deployment

    @pytest.mark.unit
    @pytest.mark.parametrize("deployment", ["LOCAL", "edge", "", "on_prem"])
    def test_deployment_rejects_unknown_values(self, deployment: str) -> None:
        # The Literal is case-sensitive; "LOCAL" must be rejected to keep
        # downstream cost-reconciliation routing unambiguous (Req 2.12).
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(_with(_valid_hardware_payload(), deployment=deployment))
        assert "deployment" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_field",
        [
            "gpu_model",
            "gpu_count",
            "vram_per_gpu_gb",
            "system_ram_gb",
            "disk_space_gb",
            "accelerator_family",
            "deployment",
        ],
    )
    def test_missing_required_field_is_rejected_with_name(self, missing_field: str) -> None:
        # Requirement 2.3 demands that every missing field be named in the
        # surfaced error so the operator can fix the profile file.
        payload = _valid_hardware_payload()
        del payload[missing_field]
        with pytest.raises(ConfigLoadError) as exc_info:
            HardwareProfile.parse(payload)
        assert missing_field in exc_info.value.fields


# ===========================================================================
# BudgetProfile (Requirement 2.2)
# ===========================================================================


class TestBudgetProfile:
    """Schema tests for :class:`BudgetProfile`.

    The model raises :class:`ConfigLoadError` for the same reason as
    :class:`HardwareProfile`: Requirement 2.3 routes both profiles through
    the same field-naming error category.
    """

    @pytest.mark.unit
    def test_parse_valid_payload_returns_populated_instance(self) -> None:
        payload = _valid_budget_payload()

        profile = BudgetProfile.parse(payload)

        assert profile.max_cost == 50.0
        assert profile.currency == "USD"
        assert profile.max_wallclock_hours == 12.0
        assert profile.cost_rate_per_gpu_hour == 0.5

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [0.0, -0.01, -100.0])
    def test_max_cost_must_be_strictly_positive(self, bad_value: float) -> None:
        # Requirement 2.2: a non-positive max cost would gate every run.
        with pytest.raises(ConfigLoadError) as exc_info:
            BudgetProfile.parse(_with(_valid_budget_payload(), max_cost=bad_value))
        assert "max_cost" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [0.0, -0.01, -1.0])
    def test_max_wallclock_hours_must_be_strictly_positive(self, bad_value: float) -> None:
        with pytest.raises(ConfigLoadError) as exc_info:
            BudgetProfile.parse(
                _with(_valid_budget_payload(), max_wallclock_hours=bad_value)
            )
        assert "max_wallclock_hours" in exc_info.value.fields

    @pytest.mark.unit
    def test_cost_rate_per_gpu_hour_zero_is_accepted(self) -> None:
        # Locally-owned hardware has zero marginal cost; the design's note
        # on BudgetProfile explicitly allows ``cost_rate_per_gpu_hour == 0``.
        profile = BudgetProfile.parse(
            _with(_valid_budget_payload(), cost_rate_per_gpu_hour=0.0)
        )
        assert profile.cost_rate_per_gpu_hour == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_value", [-0.01, -1.0, -100.0])
    def test_cost_rate_per_gpu_hour_must_be_non_negative(self, bad_value: float) -> None:
        with pytest.raises(ConfigLoadError) as exc_info:
            BudgetProfile.parse(
                _with(_valid_budget_payload(), cost_rate_per_gpu_hour=bad_value)
            )
        assert "cost_rate_per_gpu_hour" in exc_info.value.fields

    @pytest.mark.unit
    def test_currency_rejects_empty_string(self) -> None:
        with pytest.raises(ConfigLoadError) as exc_info:
            BudgetProfile.parse(_with(_valid_budget_payload(), currency=""))
        assert "currency" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_field",
        ["max_cost", "currency", "max_wallclock_hours", "cost_rate_per_gpu_hour"],
    )
    def test_missing_required_field_is_rejected_with_name(self, missing_field: str) -> None:
        payload = _valid_budget_payload()
        del payload[missing_field]
        with pytest.raises(ConfigLoadError) as exc_info:
            BudgetProfile.parse(payload)
        assert missing_field in exc_info.value.fields


# ===========================================================================
# LoRAConfig (Requirements 4.1, 4.2)
# ===========================================================================


class TestLoRAConfig:
    """Schema tests for :class:`LoRAConfig`.

    The model raises :class:`LoRAConfigInvalid` for every schema violation
    (Requirement 4.9). The presence-of-name-in-base-model check from 4.9 is
    deferred to ``LoRA_Trainer.configure`` and is covered in task 6.x; the
    schema-layer rejections covered here are: rank range, dropout range,
    bias vocabulary, and ``target_modules`` shape constraints (no empty
    list, no duplicates, every entry non-empty).
    """

    @pytest.mark.unit
    def test_parse_valid_payload_returns_populated_instance(self) -> None:
        payload = _valid_lora_payload()

        config = LoRAConfig.parse(payload)

        assert config.r == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "none"

    @pytest.mark.unit
    def test_target_modules_may_be_unset(self) -> None:
        # Requirement 4.4: ``target_modules`` defaults to None at the
        # schema layer and is resolved to ["q_proj", "v_proj"] inside
        # LoRA_Trainer once the base model is loaded.
        payload = _valid_lora_payload()
        del payload["target_modules"]
        config = LoRAConfig.parse(payload)
        assert config.target_modules is None

    @pytest.mark.unit
    @pytest.mark.parametrize("rank", [4, 128])
    def test_rank_accepts_closed_interval_endpoints(self, rank: int) -> None:
        # Requirement 4.2: rank in the closed interval [4, 128].
        config = LoRAConfig.parse(_with(_valid_lora_payload(), r=rank))
        assert config.r == rank

    @pytest.mark.unit
    @pytest.mark.parametrize("rank", [3, 0, -1, 129, 256])
    def test_rank_rejects_out_of_interval_values(self, rank: int) -> None:
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), r=rank))
        assert "r" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("alpha", [0.0, -0.01, -1.0])
    def test_alpha_must_be_strictly_positive(self, alpha: float) -> None:
        # Requirement 4.1 declares alpha as a positive number; the
        # design's data-model section reinforces ``alpha > 0``.
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), alpha=alpha))
        assert "alpha" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("dropout", [0.0, 1.0])
    def test_dropout_accepts_closed_interval_endpoints(self, dropout: float) -> None:
        # Requirement 4.1: dropout is in the closed interval [0.0, 1.0]
        # inclusive; 0.0 (no dropout) and 1.0 (drop everything) both parse.
        config = LoRAConfig.parse(_with(_valid_lora_payload(), dropout=dropout))
        assert config.dropout == dropout

    @pytest.mark.unit
    @pytest.mark.parametrize("dropout", [-0.01, 1.01, -1.0, 2.0])
    def test_dropout_rejects_out_of_interval_values(self, dropout: float) -> None:
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), dropout=dropout))
        assert "dropout" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("bias", ["none", "all", "lora_only"])
    def test_bias_accepts_each_documented_value(self, bias: str) -> None:
        # Requirement 4.1 fixes the bias vocabulary.
        config = LoRAConfig.parse(_with(_valid_lora_payload(), bias=bias))
        assert config.bias == bias

    @pytest.mark.unit
    @pytest.mark.parametrize("bias", ["NONE", "lora", "", "qkv"])
    def test_bias_rejects_unknown_values(self, bias: str) -> None:
        # The Literal is case-sensitive; "NONE" must be rejected to keep
        # peft's bias-mode dispatch unambiguous (Requirement 4.9).
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), bias=bias))
        assert "bias" in exc_info.value.fields

    @pytest.mark.unit
    def test_target_modules_rejects_empty_list(self) -> None:
        # An empty target list would yield a zero-parameter adapter; the
        # validator names the field per the *Error Handling* table.
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), target_modules=[]))
        assert "target_modules" in exc_info.value.fields

    @pytest.mark.unit
    def test_target_modules_rejects_duplicates(self) -> None:
        # Duplicates would lead to ambiguous module resolution inside
        # LoRA_Trainer, so they are rejected at parse time.
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(
                _with(_valid_lora_payload(), target_modules=["q_proj", "q_proj"])
            )
        assert "target_modules" in exc_info.value.fields

    @pytest.mark.unit
    def test_target_modules_rejects_empty_string_entry(self) -> None:
        # Each entry uses NonEmptyStr so an empty module name is rejected
        # with a path that includes the offending list index.
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(_with(_valid_lora_payload(), target_modules=["q_proj", ""]))
        # The dotted path will look like "target_modules.1" -- assert that
        # the field name appears as a prefix so the operator can locate it.
        assert any(f.startswith("target_modules") for f in exc_info.value.fields)

    @pytest.mark.unit
    @pytest.mark.parametrize("missing_field", ["r", "alpha", "dropout", "bias"])
    def test_missing_required_field_is_rejected_with_name(self, missing_field: str) -> None:
        payload = _valid_lora_payload()
        del payload[missing_field]
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            LoRAConfig.parse(payload)
        assert missing_field in exc_info.value.fields


# ===========================================================================
# DecodingParams (Requirement 7.7)
# ===========================================================================


class TestDecodingParams:
    """Schema tests for :class:`DecodingParams`.

    Decoding parameters are validated at the schema layer so that
    ``Inference_Server.generate`` (Requirement 7.7) and
    ``Evaluator.evaluate`` (Requirement 6.3) reject identical inputs.
    Violations surface as :class:`DecodingParamsInvalid` with the offending
    field named, per Property 35 in the design document.
    """

    @pytest.mark.unit
    def test_parse_valid_payload_returns_populated_instance(self) -> None:
        payload = _valid_decoding_payload()

        params = DecodingParams.parse(payload)

        assert params.temperature == 0.7
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.max_new_tokens == 256
        assert params.seed == 42

    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [0.0, 2.0])
    def test_temperature_accepts_closed_interval_endpoints(self, temperature: float) -> None:
        # Requirement 7.7: temperature in [0.0, 2.0] inclusive. 0.0 means
        # greedy decoding (used by the merge-equivalence check, Req 7.8).
        params = DecodingParams.parse(_with(_valid_decoding_payload(), temperature=temperature))
        assert params.temperature == temperature

    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [-0.01, 2.01, -1.0, 3.0])
    def test_temperature_rejects_out_of_interval_values(self, temperature: float) -> None:
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(_with(_valid_decoding_payload(), temperature=temperature))
        assert "temperature" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("top_p", [0.0, 1.0])
    def test_top_p_accepts_closed_interval_endpoints(self, top_p: float) -> None:
        # Requirement 7.7: top_p in [0.0, 1.0] inclusive.
        params = DecodingParams.parse(_with(_valid_decoding_payload(), top_p=top_p))
        assert params.top_p == top_p

    @pytest.mark.unit
    @pytest.mark.parametrize("top_p", [-0.01, 1.01, -1.0, 2.0])
    def test_top_p_rejects_out_of_interval_values(self, top_p: float) -> None:
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(_with(_valid_decoding_payload(), top_p=top_p))
        assert "top_p" in exc_info.value.fields

    @pytest.mark.unit
    def test_top_k_zero_is_accepted(self) -> None:
        # ``top_k = 0`` disables top-k sampling and is required by the
        # greedy-decoding configuration of Req 7.8 (top_k 0).
        params = DecodingParams.parse(_with(_valid_decoding_payload(), top_k=0))
        assert params.top_k == 0

    @pytest.mark.unit
    @pytest.mark.parametrize("top_k", [-1, -100])
    def test_top_k_rejects_negative_values(self, top_k: int) -> None:
        # Requirement 7.7: top_k is a non-negative integer.
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(_with(_valid_decoding_payload(), top_k=top_k))
        assert "top_k" in exc_info.value.fields

    @pytest.mark.unit
    def test_max_new_tokens_one_is_accepted(self) -> None:
        # ``max_new_tokens > 0`` -- the smallest valid value is 1.
        params = DecodingParams.parse(_with(_valid_decoding_payload(), max_new_tokens=1))
        assert params.max_new_tokens == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("max_new_tokens", [0, -1, -100])
    def test_max_new_tokens_must_be_strictly_positive(self, max_new_tokens: int) -> None:
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(
                _with(_valid_decoding_payload(), max_new_tokens=max_new_tokens)
            )
        assert "max_new_tokens" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("seed", [0, 1, -1, 2**31 - 1])
    def test_seed_accepts_any_integer(self, seed: int) -> None:
        # Requirement 7.7 declares ``seed: int`` with no range constraint;
        # negative values are accepted because numpy/torch RNGs accept them.
        params = DecodingParams.parse(_with(_valid_decoding_payload(), seed=seed))
        assert params.seed == seed

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_seed", [1.5, "42", None])
    def test_seed_rejects_non_integer_values(self, bad_seed: Any) -> None:
        # ``StrictInt`` rejects floats and strings even when they look
        # numeric (a common mistake in YAML configs).
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(_with(_valid_decoding_payload(), seed=bad_seed))
        assert "seed" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_field",
        ["temperature", "top_p", "top_k", "max_new_tokens", "seed"],
    )
    def test_missing_required_field_is_rejected_with_name(self, missing_field: str) -> None:
        payload = _valid_decoding_payload()
        del payload[missing_field]
        with pytest.raises(DecodingParamsInvalid) as exc_info:
            DecodingParams.parse(payload)
        assert missing_field in exc_info.value.fields
