"""LoRA_Trainer component (Requirement 4).

Validates a LoRA_Config, applies LoRA via the peft library to the configured
target modules of a frozen Base_Model, serializes the trained Adapter together
with its resolved configuration and base model identity, and refuses to
overwrite an existing Adapter at the same run-id path.
"""

__all__: list[str] = []
