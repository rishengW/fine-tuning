"""Cross-field validation for ``LoRA_Config`` against a loaded base model.

This module covers task 6.1 of the math-LoRA implementation plan, which
implements Requirements 4.1, 4.2, and 4.9 of the feature spec:

* Requirement 4.1 declares the field shapes (``r``, ``alpha``, ``dropout``,
  ``target_modules``, ``bias``).
* Requirement 4.2 fixes ``r in [4, 128]``.
* Requirement 4.9 states that the trainer SHALL reject the configuration
  before training begins and SHALL surface an error identifying the invalid
  field when ``r`` is outside ``[4, 128]``, ``dropout`` is outside
  ``[0.0, 1.0]``, ``bias`` is not in ``{"none", "all", "lora_only"}``, or
  ``target_modules`` contains a name that does not match any module in the
  loaded base model.

Field-shape checks (rank range, dropout range, bias vocabulary, list shape)
are implemented at the schema layer in :mod:`math_lora.types.models` so that
configuration loaders see a populated :class:`~math_lora.types.LoRAConfig`
instance (or a typed :class:`~math_lora.types.LoRAConfigInvalid` error).

The remaining check -- that every entry in ``target_modules`` matches a real
module name in the loaded base model -- requires knowledge of the loaded
transformer's parameter graph and therefore lives here in the trainer
package. The function exposed below accepts the base-model module names as
an explicit iterable so it can be unit-tested without loading a real model;
wiring to a real loaded ``transformers``/``peft`` model happens in task 6.3
when ``LoRA_Trainer.configure`` is implemented.

Design references:
    * ``.kiro/specs/math-lora-finetuning/design.md`` -- *Components and
      Interfaces* / ``LoRA_Trainer`` and *Error Handling* table entry for
      ``LoRAConfigInvalid``.
    * Property 18 in the design document covers the field-level validation
      contract this module enforces.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from math_lora.types import LoRAConfig, LoRAConfigInvalid


__all__ = ["validate"]


def validate(
    lora_config: LoRAConfig | Any,
    base_model_module_names: Iterable[str],
) -> LoRAConfig:
    """Validate a ``LoRA_Config`` against the modules of a loaded base model.

    The function combines two layers of validation:

    1. **Schema validation.** If ``lora_config`` is not already a
       :class:`~math_lora.types.LoRAConfig` instance, it is parsed via
       :meth:`LoRAConfig.parse`, which enforces ``r in [4, 128]``,
       ``dropout in [0.0, 1.0]``, ``bias in {"none", "all", "lora_only"}``,
       and the structural constraints on ``target_modules`` (no empty list,
       no duplicates, every entry non-empty). Any violation surfaces as
       :class:`~math_lora.types.LoRAConfigInvalid` with the offending field
       named in :attr:`SchemaValidationError.fields` -- this satisfies the
       schema half of Requirement 4.9.
    2. **Module-name cross-check.** When ``target_modules`` is not ``None``,
       every entry must match a real module name in
       ``base_model_module_names``. The check uses exact-match against a
       set built from the supplied iterable. This is the half of
       Requirement 4.9 that requires the trainer's view of the loaded
       base model.

    When ``target_modules`` is ``None``, the function returns the parsed
    config unchanged: Requirement 4.4 mandates that the trainer resolve an
    unset ``target_modules`` to the default ``["q_proj", "v_proj"]`` *after*
    this validation step (which is task 6.3's responsibility). Performing
    the default resolution here would prematurely commit to those names
    before the trainer knows the actual module set, so it is intentionally
    deferred.

    Args:
        lora_config: Either a validated :class:`LoRAConfig` instance or a
            raw mapping/payload to be parsed. Passing a raw payload is
            convenient for callers loading config from YAML/JSON; passing
            a parsed instance is convenient for code paths that already
            built one (e.g. tests).
        base_model_module_names: An iterable of module names exposed by the
            loaded base model. The iterable is consumed exactly once and
            stored as a ``set`` for membership tests. Module names are
            compared by exact string equality (case-sensitive); upstream
            callers in task 6.3 are expected to pass the names produced by
            ``transformers``' ``named_modules()`` walk so the trainer's
            ``peft`` configuration receives identical strings.

    Returns:
        The validated :class:`LoRAConfig`. When ``lora_config`` was already
        an instance, the same instance is returned; when a raw payload was
        supplied, the freshly-parsed instance is returned.

    Raises:
        LoRAConfigInvalid: If schema validation fails (field shape
            violation: ``r`` out of range, ``dropout`` out of range, ``bias``
            outside the documented vocabulary, or ``target_modules`` shape
            violations) or if any explicit ``target_modules`` entry is
            absent from ``base_model_module_names``. The error names the
            offending field path -- e.g. ``r``, ``dropout``, ``bias``, or
            ``target_modules.<index>`` -- and includes the bad entry value
            in its message so the operator can locate it in the
            configuration file (Requirement 4.9).

    Examples:
        >>> from math_lora.types import LoRAConfig
        >>> cfg = LoRAConfig(r=16, alpha=32, dropout=0.0,
        ...                  target_modules=["q_proj", "v_proj"], bias="none")
        >>> module_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
        >>> validate(cfg, module_names) is cfg
        True
    """

    # ---- 1. Schema validation -------------------------------------------------
    #
    # Accept either a parsed instance or a raw payload. Re-parsing a parsed
    # instance would be wasteful, so we only invoke ``parse`` when needed.
    # ``LoRAConfig.parse`` raises :class:`LoRAConfigInvalid` on failure, so
    # the schema half of Requirement 4.9 is covered by simply not catching
    # the exception here.
    if isinstance(lora_config, LoRAConfig):
        config = lora_config
    else:
        config = LoRAConfig.parse(lora_config)

    # ---- 2. target_modules cross-check ---------------------------------------
    #
    # Requirement 4.4: when ``target_modules`` is unset, the trainer applies
    # LoRA to ``q_proj`` and ``v_proj`` after loading the base model. That
    # default resolution is task 6.3's responsibility; this function only
    # rejects an *explicit* list with names that do not appear in the
    # loaded base model. Returning early here keeps the contract crisp:
    # ``validate`` is a no-op for ``target_modules is None``.
    if config.target_modules is None:
        return config

    # Materialise the iterable exactly once. Callers may pass a generator,
    # a list, or a set; we always end up with a set for O(1) membership
    # tests below. Building the set here also defends against the iterable
    # being silently exhausted by upstream code.
    available_modules = set(base_model_module_names)

    # Check each entry in declaration order so that error messages cite the
    # *first* missing entry by index. This matches the operator's mental
    # model when reading the YAML/JSON config top-to-bottom and gives
    # deterministic shrinking under property tests.
    for index, module_name in enumerate(config.target_modules):
        if module_name not in available_modules:
            field_path = f"target_modules.{index}"
            message = (
                f"LoRAConfigInvalid: target_modules entry {module_name!r} at "
                f"index {index} does not match any module in the loaded base "
                f"model"
            )
            # We construct the error directly (rather than wrapping a
            # pydantic ValidationError) because this is a cross-field
            # rejection that pydantic cannot perform without access to the
            # base-model module set. The :class:`LoRAConfigInvalid`
            # constructor accepts an explicit ``fields`` tuple and a
            # ``details`` payload, both of which we populate so downstream
            # reporting code (logs, run-manifest halts) sees the same shape
            # it would see for a schema-layer rejection.
            raise LoRAConfigInvalid(
                message,
                fields=(field_path,),
                details=(
                    {
                        "loc": ("target_modules", index),
                        "msg": (
                            f"target_modules entry {module_name!r} at "
                            f"index {index} does not match any module in "
                            f"the loaded base model"
                        ),
                        "type": "value_error.target_module_not_found",
                        "input": module_name,
                    },
                ),
            )

    return config
