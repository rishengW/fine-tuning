"""Literal types and canonical value lists for quantization and LoRA bias modes.

These are kept in a tiny, dependency-free module so that they can be imported
by any component (``Model_Selector``, ``Hardware_Budget_Planner``,
``LoRA_Trainer``, ``Inference_Server``) without pulling in the heavier
``pydantic`` model module.

References:
    * Requirement 1.2 (``Quantization_Mode`` ∈ ``{fp16, bf16, int8, nf4}``).
    * Requirement 4.1 (``bias`` ∈ ``{none, all, lora_only}``).
    * Design § Data Models / ``QuantizationMode``, ``LoRAConfig``.
"""

from __future__ import annotations

from typing import Final, Literal, get_args


# ---------------------------------------------------------------------------
# Quantization mode (Requirement 1.2)
# ---------------------------------------------------------------------------

#: The numerical precision used to load Base_Model weights during training and
#: inference. The four-element set is fixed by Requirement 1.2 and is consumed
#: by ``Model_Selector`` (VRAM estimation), ``Hardware_Budget_Planner``
#: (default-mode resolution at <24 GB VRAM, Requirement 2.4), ``LoRA_Trainer``
#: (4-bit base / bf16 adapter layout, Requirement 4.8), and
#: ``Inference_Server`` (4-bit inference path, Requirement 7.6).
QuantizationMode = Literal["fp16", "bf16", "int8", "nf4"]

#: Canonical ordered tuple of allowed ``QuantizationMode`` values.
#:
#: The ordering is deliberate: it follows the published memory-cost ordering
#: ``fp16 (2 B) -> bf16 (2 B) -> int8 (1 B) -> nf4 (0.5 B)`` per the QLoRA
#: paper (Dettmers et al. 2023) so callers that need to enumerate modes by
#: descending byte cost can iterate this tuple directly.
QUANTIZATION_MODES: Final[tuple[QuantizationMode, ...]] = get_args(QuantizationMode)

#: Bytes per parameter under each ``QuantizationMode``. Exposed here so that
#: the ``Model_Selector`` and ``Hardware_Budget_Planner`` share a single source
#: of truth (Requirement 1.3 requires VRAM coefficients to be exposed in the
#: selection report).
BYTES_PER_PARAM: Final[dict[QuantizationMode, float]] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "nf4": 0.5,
}


# ---------------------------------------------------------------------------
# LoRA bias mode (Requirement 4.1)
# ---------------------------------------------------------------------------

#: How biases are treated by the LoRA adapter. ``none`` freezes all biases,
#: ``all`` trains every bias in the base model, and ``lora_only`` trains only
#: the biases on modules that LoRA targets.
BiasMode = Literal["none", "all", "lora_only"]

#: Canonical ordered tuple of allowed ``BiasMode`` values.
BIAS_MODES: Final[tuple[BiasMode, ...]] = get_args(BiasMode)


__all__ = [
    "QuantizationMode",
    "QUANTIZATION_MODES",
    "BYTES_PER_PARAM",
    "BiasMode",
    "BIAS_MODES",
]
