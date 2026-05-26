"""Dataset_Builder component (Requirement 3).

Ingests math training data from one or more sources, normalizes records into the
Reasoning_Format, deduplicates by canonicalized problem text, splits into
training and validation sets, applies tokenizer-aware truncation that preserves
the final answer, excludes Custom_Integral_Set problems, and emits a
DatasetCard per source.

The first piece exposed here is the **source connector layer** (Requirement
3.1, task 4.1). Later tasks will add normalization (4.3), deduplication
(4.6), splitting (4.8), truncation (4.10), exclusion (4.12), and dataset-card
emission (4.14) on top of these connectors.
"""

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
    "RawRecord",
    "DatasetSource",
    "GSM8KTrainSource",
    "MATHTrainSource",
    "OpenStepByStepSource",
    "OperatorIntegralSource",
    "iter_sources",
]
