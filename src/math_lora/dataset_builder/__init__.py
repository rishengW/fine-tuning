"""Dataset_Builder component (Requirement 3).

Ingests math training data from one or more sources, normalizes records into the
Reasoning_Format, deduplicates by canonicalized problem text, splits into
training and validation sets, applies tokenizer-aware truncation that preserves
the final answer, excludes Custom_Integral_Set problems, and emits a
DatasetCard per source.

Currently exposed:

* **Source connector layer** (Requirement 3.1, task 4.1) --
  :mod:`math_lora.dataset_builder.sources`.
* **Canonicalization and deduplication** (Requirement 3.4, task 4.6) --
  :mod:`math_lora.dataset_builder.canonicalization`. Reused by task 4.8
  (validation split anti-leakage) and task 4.12 (Custom_Integral_Set
  isolation), so any change to the canonicalization rule is a
  cross-cutting change -- see that module's docstring for the
  versioning policy.

Later tasks will add normalization (4.3), splitting (4.8), truncation
(4.10), exclusion (4.12), and dataset-card emission (4.14) on top of
these primitives.
"""

from math_lora.dataset_builder.canonicalization import (
    CANONICALIZATION_FN_ID,
    CANONICALIZATION_FN_VERSION,
    TRAILING_PUNCTUATION,
    canonicalize,
    deduplicate,
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
    "CANONICALIZATION_FN_ID",
    "CANONICALIZATION_FN_VERSION",
    "TRAILING_PUNCTUATION",
    "canonicalize",
    "deduplicate",
    "RawRecord",
    "DatasetSource",
    "GSM8KTrainSource",
    "MATHTrainSource",
    "OpenStepByStepSource",
    "OperatorIntegralSource",
    "iter_sources",
]
