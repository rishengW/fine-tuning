"""Pydantic v2 domain models for the math-LoRA fine-tuning pipeline.

Each model corresponds to a data-model block in
``.kiro/specs/math-lora-finetuning/design.md`` and is constrained to match
the acceptance criteria called out in the task prompt for task 1.2:

* :class:`BaseModelCandidate`  -- Requirement 1.1
* :class:`HardwareProfile`    -- Requirement 1.2, 2.1
* :class:`BudgetProfile`      -- Requirement 2.2
* :class:`LoRAConfig`         -- Requirement 4.1, 4.2
* :class:`DecodingParams`     -- Requirement 7.7
* :class:`ReasoningRecord`    -- Requirement 3.2 (Reasoning_Format)

Models are immutable (``frozen=True``) so that values can be safely shared
between components and embedded in the ``Run_Manifest`` (Requirement 8.2)
without callers having to defensive-copy. Extra keys are forbidden so typos
in YAML/JSON configs surface as schema errors rather than silently being
discarded -- the design's *Error Handling* table requires that the offending
field be named, and that's only possible if the parser sees the typo.

Each model exposes a ``parse`` classmethod that re-raises pydantic's
:class:`pydantic.ValidationError` as the component-specific schema error
declared in :mod:`math_lora.types.errors`. Callers in later tasks
(``LoRA_Trainer``, ``Hardware_Budget_Planner``, ``Inference_Server``) use
those classmethods rather than calling ``model_validate`` directly.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from math_lora.types.errors import (
    ConfigLoadError,
    DecodingParamsInvalid,
    LoRAConfigInvalid,
    SchemaValidationError,
)
from math_lora.types.quantization import BiasMode, QuantizationMode


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _DomainModel(BaseModel):
    """Common base for every domain model in this module.

    Frozen so that values can be shared between components without
    defensive copies, and ``extra="forbid"`` so typos in configs surface
    as field errors instead of silently being dropped.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        # ``strict`` is enabled so that ``True``/``False`` are not coerced
        # into ``1``/``0`` for integer fields and so that integer literals
        # in YAML are not coerced into strings. Float fields still accept
        # int input (``7`` -> ``7.0``) which is the natural way to write
        # parameter counts in human-edited configs.
        strict=False,
    )

    # Subclasses override this to point ``parse`` at the right error type.
    _schema_error_cls: ClassVar[type[SchemaValidationError]] = SchemaValidationError

    @classmethod
    def parse(cls, data: Any) -> "Any":
        """Validate ``data`` and re-raise schema failures as the typed error.

        Returns an instance of ``cls`` on success. On a pydantic validation
        failure, raises the subclass's :attr:`_schema_error_cls` so that
        callers receive the component-specific error documented in the
        design's *Error Handling* table.
        """

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise cls._schema_error_cls.from_validation_error(exc) from exc


# Type aliases used by the field declarations below. Keeping them at the
# module level lets readers see the constraint shape next to the model.
NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]


# ---------------------------------------------------------------------------
# BaseModelCandidate (Requirement 1.1)
# ---------------------------------------------------------------------------


class BaseModelCandidate(_DomainModel):
    """A candidate base model considered by ``Model_Selector`` (Requirement 1.1).

    Field constraints follow the design's data-model section. The ``family``
    field is a free-form string rather than a closed ``Literal`` because
    Requirement 1.1 uses an open-ended set (``"qwen", "deepseek",
    "doubao", ..."``) and the spec explicitly leaves room for additional
    families in future runs.

    The ``protected_namespaces`` override clears the default ``("model_",)``
    setting from pydantic v2, which would otherwise warn about ``model_id``.
    The design explicitly uses ``model_id`` (matching HuggingFace's own
    ``AutoModel.from_pretrained(model_id=...)`` parameter), so we keep the
    field name and disable the protected-namespace check for this class.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        protected_namespaces=(),
    )

    model_id: NonEmptyStr
    revision: NonEmptyStr
    family: NonEmptyStr
    param_count_b: Annotated[StrictFloat | StrictInt, Field(gt=0)]
    license_id: NonEmptyStr
    license_allows_finetuning: StrictBool
    license_allows_adapter_redistribution: StrictBool
    license_allows_commercial_use: StrictBool
    native_context_length_tokens: Annotated[StrictInt, Field(gt=0)]
    tokenizer_family: NonEmptyStr
    baseline_gsm8k: Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=1.0)]
    baseline_math: Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# HardwareProfile (Requirements 1.2, 2.1)
# ---------------------------------------------------------------------------


class HardwareProfile(_DomainModel):
    """Declarative description of the available compute (Requirements 1.2, 2.1).

    The VRAM range ``[1, 1024]`` is the closed interval declared in
    Requirement 1.2; values outside it are rejected with a field-level error.
    """

    _schema_error_cls: ClassVar[type[SchemaValidationError]] = ConfigLoadError

    gpu_model: NonEmptyStr
    gpu_count: Annotated[StrictInt, Field(ge=1)]
    vram_per_gpu_gb: Annotated[StrictInt, Field(ge=1, le=1024)]
    system_ram_gb: Annotated[StrictInt, Field(gt=0)]
    disk_space_gb: Annotated[StrictInt, Field(gt=0)]
    accelerator_family: Literal["cuda", "rocm", "metal", "cpu"]
    deployment: Literal["local", "cloud"]


# ---------------------------------------------------------------------------
# BudgetProfile (Requirement 2.2)
# ---------------------------------------------------------------------------


class BudgetProfile(_DomainModel):
    """Monetary and time constraints for a training run (Requirement 2.2).

    ``cost_rate_per_gpu_hour`` is allowed to be exactly zero so that
    locally-owned hardware (sunk cost) can be modelled without contortion;
    ``max_cost`` and ``max_wallclock_hours`` must be strictly positive
    because a non-positive limit would gate every run.
    """

    _schema_error_cls: ClassVar[type[SchemaValidationError]] = ConfigLoadError

    max_cost: Annotated[StrictFloat | StrictInt, Field(gt=0)]
    currency: NonEmptyStr
    max_wallclock_hours: Annotated[StrictFloat | StrictInt, Field(gt=0)]
    cost_rate_per_gpu_hour: Annotated[StrictFloat | StrictInt, Field(ge=0)]


# ---------------------------------------------------------------------------
# LoRAConfig (Requirements 4.1, 4.2)
# ---------------------------------------------------------------------------


class LoRAConfig(_DomainModel):
    """LoRA hyperparameters (Requirement 4.1, 4.2).

    Note that ``target_modules`` may be ``None``: Requirement 4.4 specifies
    that ``LoRA_Trainer`` resolves an unset value to the default
    ``["q_proj", "v_proj"]`` *after* loading the base model. Resolution at
    parse time would prematurely commit to those names before the trainer
    knows the actual module set, so we keep the field optional here.
    The presence-of-name-in-base-model check (Requirement 4.9) is also
    deferred to the trainer, since this module knows nothing about loaded
    transformers.
    """

    _schema_error_cls: ClassVar[type[SchemaValidationError]] = LoRAConfigInvalid

    r: Annotated[StrictInt, Field(ge=4, le=128)]
    alpha: Annotated[StrictFloat | StrictInt, Field(gt=0)]
    dropout: Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=1.0)]
    target_modules: list[NonEmptyStr] | None = None
    bias: BiasMode

    @field_validator("target_modules")
    @classmethod
    def _target_modules_unique_and_non_empty(
        cls, value: list[str] | None
    ) -> list[str] | None:
        """Reject empty lists and duplicate names.

        An empty ``target_modules`` list is meaningless (the trainer would
        produce a zero-parameter adapter) and is more likely a config typo
        than an intentional choice. Duplicates would lead to ambiguous
        module resolution in the trainer, so we reject them here with a
        named field error per the *Error Handling* table.
        """

        if value is None:
            return value
        if len(value) == 0:
            raise ValueError("target_modules must not be an empty list")
        if len(set(value)) != len(value):
            raise ValueError("target_modules must not contain duplicate entries")
        return value


# ---------------------------------------------------------------------------
# DecodingParams (Requirement 7.7)
# ---------------------------------------------------------------------------


class DecodingParams(_DomainModel):
    """Decoding parameters for inference and evaluation (Requirement 7.7).

    The same model is used by ``Inference_Server.generate`` (Requirement 7.7)
    and by ``Evaluator.evaluate`` for baseline/post-training scoring
    (Requirement 6.3), so a single shared schema guarantees that both code
    paths reject identical inputs.
    """

    _schema_error_cls: ClassVar[type[SchemaValidationError]] = DecodingParamsInvalid

    temperature: Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=2.0)]
    top_p: Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=1.0)]
    top_k: Annotated[StrictInt, Field(ge=0)]
    max_new_tokens: Annotated[StrictInt, Field(gt=0)]
    seed: StrictInt


# ---------------------------------------------------------------------------
# ReasoningRecord (Requirement 3.2 -- Reasoning_Format)
# ---------------------------------------------------------------------------


class ReasoningRecord(_DomainModel):
    """A single training/eval record in the ``Reasoning_Format``.

    Per Requirement 3.2 every field must be non-empty and ``solution_steps``
    must contain at least one non-empty entry. Whitespace-only strings are
    rejected: the spec says "non-empty string", and a string of spaces
    carries no reasoning content.
    """

    problem: NonEmptyStr
    solution_steps: Annotated[list[NonEmptyStr], Field(min_length=1)]
    final_answer: NonEmptyStr

    @field_validator("problem", "final_answer")
    @classmethod
    def _not_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be whitespace-only")
        return value

    @field_validator("solution_steps")
    @classmethod
    def _steps_not_whitespace_only(cls, value: list[str]) -> list[str]:
        for idx, step in enumerate(value):
            if not step.strip():
                raise ValueError(
                    f"solution_steps[{idx}] must not be whitespace-only"
                )
        return value


__all__ = [
    "BaseModelCandidate",
    "HardwareProfile",
    "BudgetProfile",
    "LoRAConfig",
    "DecodingParams",
    "ReasoningRecord",
    "QuantizationMode",
    "BiasMode",
]
