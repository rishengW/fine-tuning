# math-lora

LoRA fine-tuning pipeline that adapts an open-weight LLM (Qwen, DeepSeek, or Doubao
family) for step-by-step mathematical reasoning, including indefinite and definite
integrals and symbolic derivations.

The project is operated under tight VRAM and monetary budget constraints, so
feasibility (base model selection, hardware/budget pre-flight) is a first-class
concern alongside training and evaluation.

See `.kiro/specs/math-lora-finetuning/` for the full requirements, design, and
implementation plan.

## Project layout

```
src/math_lora/            # Python package, one subpackage per component
    types/                # Shared domain types and schemas
    model_selector/       # Requirement 1
    planner/              # Requirement 2 (Hardware_Budget_Planner)
    dataset_builder/      # Requirement 3
    lora_trainer/         # Requirement 4
    pipeline/             # Requirement 5 (Training_Pipeline orchestrator)
    evaluator/            # Requirement 6
    inference_server/     # Requirement 7
    tracker/              # Requirement 8 (Experiment_Tracker)
tests/
    unit/                 # example-based unit tests
    property/             # hypothesis property-based tests (>=100 examples)
    integration/          # integration tests
```

## Setup

Requires Python 3.10-3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

`bitsandbytes` requires a CUDA-capable GPU with matching CUDA toolkit on Linux/Windows.
On platforms without CUDA, install with the `bitsandbytes` line removed from
`pyproject.toml` and skip the `nf4` quantization tests.

## Running tests

```powershell
# All tests
pytest

# Property tests only (uses the "default" hypothesis profile, max_examples=100)
pytest -m property

# Property tests in CI mode (max_examples=200)
HYPOTHESIS_PROFILE=ci pytest -m property
```
"# fine-tuning" 
