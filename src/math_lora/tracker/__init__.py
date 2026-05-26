"""Experiment_Tracker component (Requirement 8).

Records run identifier, UTC timestamps, git commit hash and dirty-tree state,
OS identifier, Python and accelerator driver versions, resolved configurations,
random seeds, dataset cards and content hashes, dependency lock, and all
metrics from Training_Pipeline and Evaluator into a Run_Manifest persisted
within 60 seconds of run completion or halt, and emits a manifest diff
comparing two runs.
"""

__all__: list[str] = []
