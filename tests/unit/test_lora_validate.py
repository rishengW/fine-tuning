"""Unit tests for :func:`math_lora.lora_trainer.validate`.

Coverage
--------

Task 6.1 of the implementation plan implements Requirements 4.1, 4.2, and
4.9 of the spec. The validation responsibility is split across two layers:

* The schema layer in :mod:`math_lora.types.models` enforces field shapes
  (``r in [4, 128]``, ``dropout in [0.0, 1.0]``, ``bias in {none, all,
  lora_only}``, and the structural rules on ``target_modules``).
* The trainer layer in :mod:`math_lora.lora_trainer.validate` adds a
  cross-field check that every entry in ``target_modules`` matches a real
  module name in the loaded base model.

This file exercises both layers through the trainer's :func:`validate`
entry point so that callers receive a single, consistent error type
(:class:`~math_lora.types.LoRAConfigInvalid`) regardless of which layer
detects the violation.

The cases below cover, per the task prompt:

1. **Schema-level rejections.** ``r`` out of ``[4, 128]``, ``dropout`` out
   of ``[0.0, 1.0]``, and an invalid ``bias`` value each raise
   :class:`LoRAConfigInvalid` and name the offending field.
2. **Cross-field rejection.** A ``target_modules`` entry that does not
   appear in the supplied module names raises :class:`LoRAConfigInvalid`,
   names ``target_modules`` (with the offending index), and includes the
   bad entry value in the error message.
3. **Happy path.** A valid config plus a matching module-name set passes.
"""

from __future__ import annotations

from typing import Any

import pytest

from math_lora.lora_trainer import validate
from math_lora.types import LoRAConfig, LoRAConfigInvalid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_lora_payload() -> dict[str, Any]:
    """Return a fully-valid raw payload for :class:`LoRAConfig`.

    Tests mutate one field at a time on a copy of this dict so each
    rejection isolates a single field violation. ``target_modules`` is
    set to a two-element list that matches the typical attention-projection
    names used by the candidate base models (Qwen, DeepSeek, Doubao).
    """

    return {
        "r": 16,
        "alpha": 32.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }


def _with(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Shallow-copy ``payload`` with ``overrides`` applied."""

    return {**payload, **overrides}


def _default_module_names() -> set[str]:
    """A representative set of module names from a transformer base model.

    Mirrors the names exposed by ``transformers``'s ``named_modules()`` walk
    on Qwen / DeepSeek / Doubao attention layers (see the design's
    *Components and Interfaces* section, ``LoRA_Trainer`` / Req 4.4).
    Tests use this set as the "loaded base model" stand-in so they do not
    need a real :class:`transformers.PreTrainedModel`.
    """

    return {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


# ===========================================================================
# Schema-level rejections (Requirements 4.1, 4.2, 4.9)
# ===========================================================================


class TestSchemaLevelRejections:
    """The trainer layer surfaces schema rejections as :class:`LoRAConfigInvalid`.

    These tests pass raw payloads (rather than parsed instances) so that the
    schema is exercised through ``validate``. The trainer must not catch or
    transform schema errors -- callers see the same typed exception they
    would see if they had called :meth:`LoRAConfig.parse` directly.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("rank", [3, 0, -1, 129, 256])
    def test_rank_out_of_interval_raises_naming_field(self, rank: int) -> None:
        # Requirement 4.2: ``r in [4, 128]``. Values just outside the
        # closed interval as well as far-out values must be rejected with
        # the field ``r`` named.
        payload = _with(_valid_lora_payload(), r=rank)
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())
        assert "r" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("rank", [4, 128])
    def test_rank_at_closed_interval_endpoints_passes(self, rank: int) -> None:
        # Requirement 4.2 is a closed interval: 4 and 128 are valid.
        config = validate(_with(_valid_lora_payload(), r=rank), _default_module_names())
        assert config.r == rank

    @pytest.mark.unit
    @pytest.mark.parametrize("dropout", [-0.01, 1.01, -1.0, 2.0])
    def test_dropout_out_of_interval_raises_naming_field(
        self, dropout: float
    ) -> None:
        # Requirement 4.1 / 4.9: dropout in ``[0.0, 1.0]``; anything else
        # is rejected with the field ``dropout`` named.
        payload = _with(_valid_lora_payload(), dropout=dropout)
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())
        assert "dropout" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("dropout", [0.0, 1.0])
    def test_dropout_at_closed_interval_endpoints_passes(
        self, dropout: float
    ) -> None:
        # 0.0 (no dropout) and 1.0 (drop everything) are both inside the
        # closed interval and must be accepted.
        config = validate(
            _with(_valid_lora_payload(), dropout=dropout), _default_module_names()
        )
        assert config.dropout == dropout

    @pytest.mark.unit
    @pytest.mark.parametrize("bias", ["NONE", "lora", "", "qkv", "linear"])
    def test_bias_outside_vocabulary_raises_naming_field(self, bias: str) -> None:
        # Requirement 4.1 / 4.9: bias in the closed set ``{"none", "all",
        # "lora_only"}``. The Literal is case-sensitive -- ``"NONE"`` must
        # be rejected so peft's bias-mode dispatch stays unambiguous.
        payload = _with(_valid_lora_payload(), bias=bias)
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())
        assert "bias" in exc_info.value.fields

    @pytest.mark.unit
    @pytest.mark.parametrize("bias", ["none", "all", "lora_only"])
    def test_bias_at_each_documented_value_passes(self, bias: str) -> None:
        config = validate(
            _with(_valid_lora_payload(), bias=bias), _default_module_names()
        )
        assert config.bias == bias


# ===========================================================================
# target_modules cross-check (Requirement 4.9)
# ===========================================================================


class TestTargetModulesCrossCheck:
    """Cross-field check against the loaded base model's module names.

    The schema layer cannot perform this check because it has no access to
    the base model; the trainer layer therefore receives an explicit
    iterable of module names and is responsible for surfacing
    :class:`LoRAConfigInvalid` with the offending field path *and* the bad
    entry value (Requirement 4.9).
    """

    @pytest.mark.unit
    def test_unknown_target_module_raises_naming_field_and_entry(self) -> None:
        # ``"x_proj"`` is not in the supplied module set, so validation
        # must reject the config with a field path that points at the
        # offending index and an error message that contains the bad
        # entry value so the operator can locate it in their config file.
        payload = _with(
            _valid_lora_payload(), target_modules=["q_proj", "x_proj"]
        )
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())

        # Field path includes ``target_modules`` and the offending index.
        assert any(
            field.startswith("target_modules") for field in exc_info.value.fields
        )
        assert "target_modules.1" in exc_info.value.fields
        # The error message must name the bad entry value so a search of
        # the config file lands directly on the offending line.
        assert "'x_proj'" in str(exc_info.value)

    @pytest.mark.unit
    def test_first_unknown_entry_is_reported_when_multiple_are_missing(
        self,
    ) -> None:
        # When more than one entry is missing we report the *first* one in
        # declaration order. This gives operators a deterministic path to
        # the source of the error and matches the order they would read in
        # a top-to-bottom config file.
        payload = _with(
            _valid_lora_payload(),
            target_modules=["q_proj", "first_missing", "second_missing"],
        )
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())

        assert "target_modules.1" in exc_info.value.fields
        assert "'first_missing'" in str(exc_info.value)
        # The second missing entry must not appear in the field list,
        # confirming that we stop at the first violation.
        assert "target_modules.2" not in exc_info.value.fields

    @pytest.mark.unit
    def test_validation_is_case_sensitive(self) -> None:
        # Module-name comparison uses exact string equality so an upstream
        # typo like ``"Q_proj"`` (uppercase Q) is caught even though the
        # lowercase ``"q_proj"`` is present in the base model. This is
        # important because ``transformers``' ``named_modules()`` walk
        # produces canonical lowercase names; matching loosely would let
        # bad configs slip through.
        payload = _with(
            _valid_lora_payload(), target_modules=["Q_proj", "v_proj"]
        )
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, _default_module_names())
        assert "target_modules.0" in exc_info.value.fields
        assert "'Q_proj'" in str(exc_info.value)

    @pytest.mark.unit
    def test_empty_module_name_set_rejects_any_target(self) -> None:
        # If the caller supplies an empty module-name iterable (e.g.
        # because they forgot to pass the loaded base model), every
        # explicit ``target_modules`` entry will be missing. The first
        # entry is reported.
        payload = _valid_lora_payload()
        with pytest.raises(LoRAConfigInvalid) as exc_info:
            validate(payload, set())
        assert "target_modules.0" in exc_info.value.fields
        assert "'q_proj'" in str(exc_info.value)


# ===========================================================================
# Happy path (Requirement 4.1)
# ===========================================================================


class TestHappyPath:
    """A valid config plus matching module names passes validation."""

    @pytest.mark.unit
    def test_valid_config_with_matching_modules_returns_parsed_instance(
        self,
    ) -> None:
        # The function returns a populated :class:`LoRAConfig` so callers
        # in task 6.3 can pipe it straight into ``peft.LoraConfig`` without
        # re-parsing.
        config = validate(_valid_lora_payload(), _default_module_names())
        assert isinstance(config, LoRAConfig)
        assert config.r == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "none"

    @pytest.mark.unit
    def test_validate_accepts_already_parsed_instance(self) -> None:
        # Callers that already hold a :class:`LoRAConfig` should not be
        # forced to round-trip through a dict. The function returns the
        # *same* object so identity-aware downstream code (caches, run
        # manifests) sees no change.
        cfg = LoRAConfig(
            r=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        result = validate(cfg, _default_module_names())
        assert result is cfg

    @pytest.mark.unit
    def test_validate_accepts_iterable_of_module_names(self) -> None:
        # The signature documents an iterable so callers can pass a list,
        # set, or generator. We exercise a generator here to make sure the
        # iterable is consumed exactly once and not treated like a sized
        # collection by mistake.
        def module_name_iter():
            yield from _default_module_names()

        config = validate(_valid_lora_payload(), module_name_iter())
        assert config.target_modules == ["q_proj", "v_proj"]

    @pytest.mark.unit
    def test_target_modules_unset_is_valid_regardless_of_modules(self) -> None:
        # Requirement 4.4 defers default resolution (``["q_proj",
        # "v_proj"]``) to the trainer's ``configure`` step (task 6.3).
        # ``validate`` must therefore accept ``target_modules=None`` even
        # when the supplied module-name set is empty -- the
        # cross-validation is a no-op until the trainer resolves the
        # default against the actual loaded base model.
        payload = _valid_lora_payload()
        del payload["target_modules"]
        config = validate(payload, set())
        assert config.target_modules is None

    @pytest.mark.unit
    def test_target_modules_subset_of_module_names_passes(self) -> None:
        # Validation only requires that every entry in ``target_modules``
        # is present in the module-name set; the converse (every module
        # name appearing in target_modules) is intentionally not required
        # because LoRA only adapts a subset of modules by design.
        payload = _with(_valid_lora_payload(), target_modules=["q_proj"])
        config = validate(payload, _default_module_names())
        assert config.target_modules == ["q_proj"]
