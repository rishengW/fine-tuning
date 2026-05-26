"""Shared domain types and schemas used across all components.

This package centralizes the data models and exception classes that every
other component depends on:

* Pydantic v2 models for the configuration and record schemas defined in
  the design's *Data Models* section (``BaseModelCandidate``,
  ``HardwareProfile``, ``BudgetProfile``, ``LoRAConfig``, ``DecodingParams``,
  ``ReasoningRecord``).
* Literal types for ``QuantizationMode`` and ``BiasMode`` plus the
  associated canonical value tuples.
* Field-level validation errors (``SchemaValidationError`` and the three
  category-specific subclasses ``LoRAConfigInvalid``, ``ConfigLoadError``,
  ``DecodingParamsInvalid``) that name the offending field, as required by
  the design's *Error Handling* table.
"""

from math_lora.types.errors import (
    ConfigLoadError,
    DecodingParamsInvalid,
    LoRAConfigInvalid,
    MathLoRAError,
    SchemaValidationError,
)
from math_lora.types.models import (
    BaseModelCandidate,
    BudgetProfile,
    DecodingParams,
    HardwareProfile,
    LoRAConfig,
    ReasoningRecord,
)
from math_lora.types.quantization import (
    BIAS_MODES,
    BYTES_PER_PARAM,
    QUANTIZATION_MODES,
    BiasMode,
    QuantizationMode,
)

__all__ = [
    # Models
    "BaseModelCandidate",
    "HardwareProfile",
    "BudgetProfile",
    "LoRAConfig",
    "DecodingParams",
    "ReasoningRecord",
    # Quantization / bias literals and tables
    "QuantizationMode",
    "QUANTIZATION_MODES",
    "BYTES_PER_PARAM",
    "BiasMode",
    "BIAS_MODES",
    # Errors
    "MathLoRAError",
    "SchemaValidationError",
    "LoRAConfigInvalid",
    "ConfigLoadError",
    "DecodingParamsInvalid",
]
