"""Inference_Server component (Requirement 7).

Loads a Base_Model and zero or more named Adapters (validating each Adapter's
recorded base_model_id and revision), responds to math prompts in the
Reasoning_Format, switches the active Adapter without unloading the base,
honors no_adapter for base-only inference, and supports merging an Adapter
into a standalone artifact with token-equality verification on a fixed prompt
set.
"""

__all__: list[str] = []
