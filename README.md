# math-lora

Minimal LoRA fine-tuning for mathematical reasoning. One script downloads and
formats public math datasets, another trains a LoRA adapter on top of a small
open-weight chat model.

## Layout

```
src/math_lora/
    build_dataset.py    # download + format public datasets -> jsonl
    train.py            # LoRA fine-tuning script
data/
    train.jsonl         # training set (chat-style messages)
    val.jsonl           # validation set
```

## Setup

Requires Python 3.10-3.12. A CUDA-capable GPU is recommended for practical
training speed; the code falls back to CPU/fp32 otherwise.

The commands differ slightly per shell. Pick the row that matches your
environment and use that style for every command in this README.

**Linux / macOS (bash, zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If PowerShell refuses to run the activation script, allow it once per session
with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

**Windows cmd:**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
```

## Data format

Each line of `data/train.jsonl` and `data/val.jsonl` is a JSON object with a
`messages` field following the OpenAI chat format:

```json
{"messages": [
    {"role": "user", "content": "What is 2 + 3?"},
    {"role": "assistant", "content": "Step 1: add the numbers. 2 + 3 = 5.\nFinal answer: 5"}
]}
```

The repo ships with a tiny hand-written seed set (10 train + 3 val) so you
can sanity-check the training loop without downloading anything.

## Build a real dataset

To pull a real, diverse training set from public Hugging Face datasets, run
one of the following depending on your shell. Each form passes the same
arguments; only the line-continuation character differs.

**Linux / macOS (bash, zsh)** — backslash continues a line:

```bash
python -m math_lora.build_dataset \
    --train-size 3000 \
    --val-size 300
```

**Windows PowerShell** — backtick continues a line:

```powershell
python -m math_lora.build_dataset `
    --train-size 3000 `
    --val-size 300
```

**Windows cmd** — caret continues a line:

```cmd
python -m math_lora.build_dataset ^
    --train-size 3000 ^
    --val-size 300
```

Or, on any shell, put it all on one line:

```text
python -m math_lora.build_dataset --train-size 3000 --val-size 300
```

What this does:

- Downloads [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (~7.5k clean
  grade-school word problems) and rewrites the trailing `#### 42` answer
  marker as `Final answer: 42` to match the seed format.
- Streams a slice of [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
  (~860k competition-style problem/solution pairs) for broader coverage,
  without downloading the full corpus to disk.
- Drops records whose total length exceeds `--max-chars` (default 1500) so
  responses fit comfortably in the training context.
- Defaults to a 70/30 GSM8K/NuminaMath mix at 3k train + 300 val. The GSM8K
  lean keeps difficulty appropriate for a small base model; the NuminaMath
  slice prevents the model from only seeing one style.

CLI knobs: `--train-size`, `--val-size`, `--gsm8k-ratio`, `--max-chars`,
`--seed`, `--train-out`, `--val-out`. The script overwrites the output files.

## Train

**Linux / macOS (bash, zsh):**

```bash
python -m math_lora.train \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --output-dir outputs/adapter
```

**Windows PowerShell:**

```powershell
python -m math_lora.train `
    --base-model Qwen/Qwen2.5-0.5B-Instruct `
    --train-file data/train.jsonl `
    --val-file data/val.jsonl `
    --output-dir outputs/adapter
```

**Windows cmd:**

```cmd
python -m math_lora.train ^
    --base-model Qwen/Qwen2.5-0.5B-Instruct ^
    --train-file data/train.jsonl ^
    --val-file data/val.jsonl ^
    --output-dir outputs/adapter
```

Defaults: `Qwen/Qwen2.5-0.5B-Instruct` base, LoRA rank 16, alpha 32, dropout
0.05, lr 2e-4, batch size 1 with 8-step grad accumulation, 3 epochs, bf16 on
GPU. Run `python -m math_lora.train --help` for the full flag list.

The trained adapter and tokenizer are saved under `--output-dir` and can be
loaded later with `peft.PeftModel.from_pretrained(base_model, output_dir)`.

## Tuning notes for small models

- **Diversity beats volume.** A few thousand varied examples generalize better
  than tens of examples repeated for many epochs.
- **Watch validation loss.** If it stops dropping (or rises) while training
  loss keeps falling, lower the rank (`--lora-r 8`), raise dropout
  (`--lora-dropout 0.1`), or reduce epochs (`--num-epochs 1`).
- **Length matters.** Long olympiad-style traces are hard for a 0.5B model
  to fit; lowering `--max-chars` in `build_dataset` produces an easier set.

## Shell quick reference

| Operation | bash / zsh (Linux, macOS) | PowerShell (Windows) | cmd (Windows) |
| --- | --- | --- | --- |
| Activate venv | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` |
| Deactivate venv | `deactivate` | `deactivate` | `deactivate` |
| Line continuation | `\` (backslash) | `` ` `` (backtick) | `^` (caret) |
| Set env var (one command) | `VAR=value python ...` | `$env:VAR='value'; python ...` | `set VAR=value && python ...` |
| Path separator in args | `data/train.jsonl` | `data/train.jsonl` or `data\train.jsonl` | `data\train.jsonl` |

Forward slashes in file paths work on every platform when passed as Python
arguments, so the example commands above are portable as long as you use the
right line-continuation character (or keep everything on one line).
