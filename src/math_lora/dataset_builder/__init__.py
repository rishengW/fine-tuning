"""Dataset_Builder component (Requirement 3).

Ingests math training data from one or more sources, normalizes records into the
Reasoning_Format, deduplicates by canonicalized problem text, splits into
training and validation sets, applies tokenizer-aware truncation that preserves
the final answer, excludes Custom_Integral_Set problems, and emits a
DatasetCard per source.

Currently exposed:

* **Source connectors** (Requirement 3.1, task 4.1): GSM8K, MATH, an
  open-source step-by-step corpus, and operator-supplied integral pairs.
* **Reasoning_Format normalization and per-record validation**
  (Requirement 3.2, 3.3, 3.6, task 4.3): ``normalize_record`` and
  ``normalize_records`` convert raw connector output into strict
  :class:`~math_lora.types.ReasoningRecord` instances and count rejections
  by reason.

Later tasks add deduplication (4.6), train/val splitting (4.8),
tokenizer-aware truncation (4.10), Custom_Integral_Set exclusion (4.12),
and dataset-card emission (4.14) on top of these pieces.
"""

from math_lora.dataset_builder.normalization import (
    ALL_REJECTION_REASONS,
    REASON_EMPTY_FINAL_ANSWER,
    REASON_EMPTY_PROBLEM,
    REASON_EMPTY_SOLUTION_STEPS,
    NormalizationResult,
    RejectedRecord,
    normalize_record,
    normalize_records,
)
from math_lora.dataset_builder.sources import (
    DatasetSource,
    GSM8KTrainSource,
    MATHTrainSource,
    OpenStepByStepSource,
    OperatorIntegralSource,
    RawRecord,
    iter_sources,
)

__all__ = [
    # Source connectors (Requirement 3.1)
    "RawRecord",
    "DatasetSource",
    "GSM8KTrainSource",
    "MATHTrainSource",
    "OpenStepByStepSource",
    "OperatorIntegralSource",
    "iter_sources",
    # Normalization (Requirement 3.2, 3.3, 3.6)
    "REASON_EMPTY_PROBLEM",
    "REASON_EMPTY_SOLUTION_STEPS",
    "REASON_EMPTY_FINAL_ANSWER",
    "ALL_REJECTION_REASONS",
    "RejectedRecord",
    "NormalizationResult",
    "normalize_record",
    "normalize_records",
]
