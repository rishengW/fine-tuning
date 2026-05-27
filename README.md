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

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Linux / macOS (bash or zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On a fresh Ubuntu/Debian box you may also need the Python venv package and
build basics before the first `python3 -m venv`:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential
```

For CUDA on Linux, install a `torch` wheel that matches your driver before
`pip install -e .` if the default CPU/CUDA wheel from PyPI doesn't fit your
setup. Example for CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
pip install -e .
```

Verify the GPU is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> **Command syntax note.** The examples below use PowerShell's backtick
> (`` ` ``) for line continuation. On Linux/macOS, replace each trailing
> backtick with a backslash (`\`), or just put the whole command on one line.

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

To pull a real, diverse training set from public Hugging Face datasets, run:

```powershell
python -m math_lora.build_dataset `
    --train-size 3000 `
    --val-size 300
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

```powershell
python -m math_lora.train `
    --base-model Qwen/Qwen2.5-0.5B-Instruct `
    --train-file data/train.jsonl `
    --val-file data/val.jsonl `
    --output-dir outputs/adapter
```

Defaults: `Qwen/Qwen2.5-0.5B-Instruct` base, LoRA rank 16, alpha 32, dropout
0.05, lr 2e-4, batch size 1 with 8-step grad accumulation, 3 epochs, bf16 on
GPU. Run `python -m math_lora.train --help` for the full flag list.

The trained adapter and tokenizer are saved under `--output-dir` and can be
loaded later with `peft.PeftModel.from_pretrained(base_model, output_dir)`.

## Test before/after

To check whether fine-tuning actually moved the needle, run the side-by-side
comparison. It evaluates the base model alone, then the base model with the
adapter attached, on a fixed prompt set covering GSM8K-style word problems,
short calculus, and a couple of algebra items.

```powershell
# Run both passes and print a comparison table
python -m math_lora.compare --adapter outputs/adapter
```

This writes `results/before.json` and `results/after.json` (full responses
plus per-prompt correctness) and prints a summary like:

```
prompt          | expected       | before             | after              | Δ
----------------+----------------+--------------------+--------------------+------
gsm-apples      | 27             | 24                 | 27                 | +1
gsm-train       | 160            | 160 miles          | 160                | =
calc-int-power  | 2x^3 + C       | x^3 + C            | 2x^3 + C           | +1
...
before: 4/10 (40.0%)    after: 8/10 (80.0%)    delta: +40.0%
flips: +5 better, -1 worse
```

You can also evaluate one model at a time with `math_lora.evaluate` (e.g.
to log a baseline before training, then re-run with `--adapter` later):

```powershell
# Baseline (no adapter)
python -m math_lora.evaluate --output results/before.json

# After training
python -m math_lora.evaluate --adapter outputs/adapter --output results/after.json

# Compare two existing reports without re-running the model
python -m math_lora.compare --report-only `
    --before-report results/before.json `
    --after-report  results/after.json
```

The prompt set lives in `src/math_lora/eval_prompts.py`. Each prompt has an
expected final answer and optional aliases; the runner extracts the model's
stated final answer (looking for `Final answer:`, `Answer:`, `\boxed{...}`,
or the `#### ...` marker) and grades it with light normalization. The full
generated response is stored in the JSON report for manual review when the
auto-grade looks suspicious.

## Tuning notes for small models

- **Diversity beats volume.** A few thousand varied examples generalize better
  than tens of examples repeated for many epochs.
- **Watch validation loss.** If it stops dropping (or rises) while training
  loss keeps falling, lower the rank (`--lora-r 8`), raise dropout
  (`--lora-dropout 0.1`), or reduce epochs (`--num-epochs 1`).
- **Length matters.** Long olympiad-style traces are hard for a 0.5B model
  to fit; lowering `--max-chars` in `build_dataset` produces an easier set.
