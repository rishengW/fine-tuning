# math-lora

LoRA / QLoRA fine-tuning pipeline for mathematical reasoning, built around
small open-weight chat models (Qwen 0.5B / 7B). Config-driven training,
GSM8K-based evaluation with before/after lift reporting, optional W&B
tracking, a thin inference HTTP server, and a reproducible Docker
environment.

```
fine-tuning/
├── configs/                # YAML run configs (0.5B LoRA, 7B QLoRA)
├── data/                   # JSONL training data (chat-message format)
├── docker/                 # Reproducible CUDA training image
├── docs/                   # architecture.md, runbook.md, eval_results.md
├── scripts/                # Thin shell entrypoints
├── src/math_lora/          # Python package
│   ├── build_dataset.py    # GSM8K + NuminaMath-CoT -> data/*.jsonl
│   ├── config.py           # Pydantic config models + override layer
│   ├── evaluate.py         # Eval runner (curated + GSM8K test split)
│   ├── evaluation/         # Answer extraction, matchers, GSM8K loader, prompts
│   │   └── eval_prompts.py # Curated before/after comparison prompts
│   ├── logging_utils.py    # Stdlib logging + W&B wrapper
│   ├── report_diff.py      # Compare two eval reports, print lift
│   ├── serve.py            # Minimal HTTP inference endpoint
│   └── train.py            # Config-driven LoRA / QLoRA trainer
├── tests/unit/             # Pure-python unit tests (no GPU needed)
├── .github/workflows/      # CI: lint + unit tests
├── Makefile                # `make train`, `make eval`, `make lint`, ...
└── pyproject.toml          # Pinned deps, optional `[dev]` and `[tracking]` extras
```

## Shell quick reference

This project supports any modern shell. Where commands differ, sections
below show three variants. The table below summarizes the differences
that trip people up most often.

| Topic | bash / zsh (Linux, macOS) | PowerShell (Windows) | cmd (Windows) |
| --- | --- | --- | --- |
| Activate venv | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` |
| Deactivate venv | `deactivate` | `deactivate` | `deactivate` |
| Line continuation | `\` (backslash) | `` ` `` (backtick) | `^` (caret) |
| Set env var (one cmd) | `VAR=value cmd ...` | `$env:VAR='value'; cmd ...` | `set VAR=value && cmd ...` |
| Make-style overrides | `make eval LIMIT=500` | `$env:LIMIT='500'; make eval` | `set LIMIT=500 && make eval` |
| `$PWD` in commands | `$PWD` | `${PWD}` | `%cd%` |

If you do not have `make` installed (typical on Windows), every `make`
target maps to a one-line `python -m ...` command - see the **Without
make** subsections below each command block.

## Environment setup

Requires Python 3.10-3.12. A CUDA-capable GPU is strongly recommended for
practical training speed; the 7B QLoRA config additionally requires
`bitsandbytes` (Linux/Windows only) and >=20 GB VRAM.

**Linux / macOS (bash, zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If PowerShell refuses to run the activation script, allow it once per
session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows cmd:**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

## Quickstart

With `make` available (Linux, macOS, or Windows with `make` installed):

```bash
make install          # pip install -e ".[dev]"
make dataset          # download GSM8K + NuminaMath-CoT -> data/
make train            # LoRA fine-tune Qwen2.5-0.5B-Instruct
make eval             # baseline vs fine-tuned on GSM8K test (200 samples)
```

Without `make`, run the equivalent Python module commands directly:

**Linux / macOS:**

```bash
pip install -e ".[dev]"
python -m math_lora.build_dataset --train-size 3000 --val-size 300
python -m math_lora.train --config configs/qwen-0.5b-lora.yaml
python -m math_lora.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct --suite gsm8k --limit 200 --report-out reports/baseline.json
python -m math_lora.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen-0.5b-lora --suite gsm8k --limit 200 --report-out reports/finetuned.json
python -m math_lora.report_diff reports/baseline.json reports/finetuned.json
```

**Windows PowerShell / cmd:** same commands, identical syntax (Python is
shell-agnostic). Use the platform-specific line continuation only if you
choose to break a command across lines.

## Data format

Each line of `data/train.jsonl` and `data/val.jsonl` is a JSON object
with a `messages` field following the OpenAI chat format:

```json
{"messages": [
    {"role": "user", "content": "What is 2 + 3?"},
    {"role": "assistant", "content": "Step 1: add the numbers. 2 + 3 = 5.\nFinal answer: 5"}
]}
```

The repo ships with a tiny hand-written seed set (10 train + 3 val) so
the training loop is runnable without downloading anything. Use
`make dataset` (or the `python -m math_lora.build_dataset` command above)
to replace it with a real ~3k-example mix from GSM8K and NuminaMath-CoT
(length-filtered for a small base model).

## Train

Runs are driven by YAML configs under `configs/`. The default targets a
0.5B model on a single consumer GPU; a separate 7B QLoRA config is
provided for larger setups.

The same command split across lines, in three shells:

**Linux / macOS (bash, zsh)** - backslash continues a line:

```bash
python -m math_lora.train \
    --config configs/qwen-0.5b-lora.yaml \
    --override training.num_epochs=1 \
    --override lora.r=4
```

**Windows PowerShell** - backtick continues a line:

```powershell
python -m math_lora.train `
    --config configs/qwen-0.5b-lora.yaml `
    --override training.num_epochs=1 `
    --override lora.r=4
```

**Windows cmd** - caret continues a line:

```cmd
python -m math_lora.train ^
    --config configs/qwen-0.5b-lora.yaml ^
    --override training.num_epochs=1 ^
    --override lora.r=4
```

Or on any shell, single-line:

```text
python -m math_lora.train --config configs/qwen-0.5b-lora.yaml --override training.num_epochs=1 --override lora.r=4
```

For the 7B QLoRA config (needs >=20 GB VRAM + `bitsandbytes`):

```bash
make train CONFIG=configs/qwen-7b-qlora.yaml ADAPTER=outputs/qwen-7b-qlora
```

Without `make`:

```bash
python -m math_lora.train --config configs/qwen-7b-qlora.yaml
```

Each run persists the resolved config to `outputs/<run>/run_config.json`
alongside the adapter, so any later replay reuses identical
hyperparameters.

## Evaluate

With `make` (env-style overrides differ per shell):

**Linux / macOS:**

```bash
make eval                                     # GSM8K test, 200 samples
make eval LIMIT=1000 ADAPTER=outputs/custom   # larger sweep, custom adapter
```

**Windows PowerShell:**

```powershell
make eval
$env:LIMIT='1000'; $env:ADAPTER='outputs/custom'; make eval
```

**Windows cmd:**

```cmd
make eval
set LIMIT=1000 && set ADAPTER=outputs/custom && make eval
```

Without `make`, two `python -m math_lora.evaluate` runs followed by a
diff:

```bash
python -m math_lora.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct --suite gsm8k --limit 200 --report-out reports/baseline.json
python -m math_lora.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen-0.5b-lora --suite gsm8k --limit 200 --report-out reports/finetuned.json
python -m math_lora.report_diff reports/baseline.json reports/finetuned.json
```

`make eval` runs three steps: baseline (no adapter), fine-tuned (with
adapter), then prints the per-category accuracy delta via
`math_lora.report_diff`. The JSON reports include every prompt's full
response, extracted answer, and correctness flag, which makes failure
analysis straightforward.

Failure inspection with `jq` (Linux, macOS, or any shell with jq):

```bash
jq '.records[] | select(.correct == false) | {id, expected_answer, extracted_answer}' reports/finetuned.json | head -50
```

PowerShell users without `jq` can use `ConvertFrom-Json`:

```powershell
$report = Get-Content reports/finetuned.json -Raw | ConvertFrom-Json
$report.records | Where-Object { -not $_.correct } | Select-Object id, expected_answer, extracted_answer | Select-Object -First 50
```

For the curated 10-prompt suite (calculus + algebra + word problems):

```bash
python -m math_lora.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen-0.5b-lora --suite curated --report-out reports/curated.json
```

## Serve

`serve.py` exposes a minimal `POST /generate` endpoint backed by the
saved adapter. It is intentionally simple - for production-grade serving
prefer vLLM or TGI, both of which load LoRA adapters with a single flag.

Start the server (any shell):

```bash
python -m math_lora.serve --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen-0.5b-lora --port 8000
```

Hit it - the `curl` invocation differs per shell:

**Linux / macOS / Git Bash:**

```bash
curl -s http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 7 * 8?"}]}'
```

**Windows PowerShell** - prefer `Invoke-RestMethod` (avoids quoting hell
because PowerShell has its own `curl` alias that behaves differently):

```powershell
$body = @{ messages = @(@{ role = 'user'; content = 'What is 7 * 8?' }) } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -ContentType 'application/json' -Body $body
```

**Windows cmd:**

```cmd
curl -s http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 7 * 8?\"}]}"
```

## Tracking

W&B is opt-in. Set `tracking.enabled=true` in the YAML or override at the
CLI:

```bash
python -m math_lora.train --config configs/qwen-0.5b-lora.yaml --override tracking.enabled=true
```

Then log in once per machine:

```bash
wandb login
```

If `wandb` is not installed, the run continues with stdout-only logging
and a warning.

## Development

```bash
make lint                         # ruff
make typecheck                    # mypy
make test                         # pytest tests/unit (no GPU needed)
```

Without `make`:

```bash
python -m ruff check src
python -m mypy src
python -m pytest tests/unit
```

The unit tests cover the answer-extraction logic and the config layer,
which are the riskiest pure-python pieces. The torch-dependent code is
exercised via integration smoke runs (`make train` + `make eval`); CI
runs only the pure-python suite to stay fast and free of GPU runners.

## Docker

The build command is the same on every shell; only the volume-mount
syntax differs because of how each shell expands the working-directory
variable.

Build (any shell):

```bash
docker build -t math-lora -f docker/Dockerfile .
```

Run with outputs persisted - **Linux / macOS / Git Bash:**

```bash
docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" math-lora
```

**Windows PowerShell:**

```powershell
docker run --rm --gpus all -v "${PWD}/outputs:/app/outputs" math-lora
```

**Windows cmd:**

```cmd
docker run --rm --gpus all -v "%cd%/outputs:/app/outputs" math-lora
```

The image is layered so source edits do not invalidate the dependency
layer. `outputs/`, `reports/`, and the HF cache are exposed as volumes.

## Tuning notes for small models

- **Diversity beats volume.** A few thousand varied examples generalize
  better than tens repeated for many epochs.
- **Watch validation loss.** If it stops dropping while train loss keeps
  falling, lower the rank (`--override lora.r=4`), raise dropout
  (`--override lora.dropout=0.1`), or reduce epochs.
- **Length matters.** Long olympiad-style traces are hard for 0.5B; lower
  `--max-chars` in `build_dataset` for an easier set.
- **Rank does not scale with model size.** It scales with how far the
  task is from pretraining. See `docs/architecture.md` for the rationale.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - design decisions and
  why the project is split this way.
- [`docs/runbook.md`](docs/runbook.md) - step-by-step from clone to
  benchmarked adapter.
- [`docs/eval_results.md`](docs/eval_results.md) - template for recording
  baseline vs fine-tuned numbers per run.
