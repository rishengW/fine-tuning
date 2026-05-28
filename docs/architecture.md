# Architecture

## Overview

`math-lora` is a small, opinionated LoRA fine-tuning pipeline for math
reasoning. It splits responsibilities into three runnable stages:

1. **`build_dataset`** - pulls public datasets (GSM8K, NuminaMath-CoT),
   normalizes them into a uniform chat-message JSONL schema, and writes
   them under `data/`.
2. **`train`** - reads a YAML config, applies a LoRA (or QLoRA) adapter
   to a base model, runs supervised fine-tuning, and persists the
   adapter + the resolved config under `outputs/<run-name>/`.
3. **`evaluate`** - loads a base model (with or without an adapter),
   generates completions for a chosen suite (curated prompts or the
   GSM8K test split), and writes a JSON report. `report_diff` compares
   two such reports and prints the lift.

A separate `serve` module provides a minimal HTTP endpoint for ad-hoc
inference against a saved adapter.

## Why YAML configs (not argparse)

Hand-typed CLI flags do not scale once you have multiple base models,
ranks, dataset slices, and run names to track. Two concrete benefits:

- **Reproducibility.** The full effective config is serialized verbatim
  into `outputs/<run>/run_config.json`, so any future replay reuses
  identical hyperparameters without depending on shell history.
- **Validation.** `RunConfig` (Pydantic) catches typos and bad types at
  load time with a single readable error, instead of mysterious
  `KeyError`s after model load.

CLI overrides (`--override training.num_epochs=1`) keep one-off
experimentation fast without forking the YAML.

## Why split GSM8K and NuminaMath in `build_dataset`

A 0.5B base model has limited capacity. Olympiad-grade traces in
NuminaMath are too long and too hard for it to fit; pure GSM8K is too
narrow. The 70/30 default mix gives:

- **Difficulty floor** - GSM8K trains the chain-of-thought structure on
  easy problems the model can actually solve.
- **Coverage** - the NuminaMath slice exposes the model to algebra,
  calculus, and contest-style framing without dominating the loss.

`--max-chars` filters out records that would not fit in the training
context, which would otherwise be silently truncated and lose the final
answer.

## Why a separate `evaluation/` subpackage

Evaluation logic is shared between two callers (the curated harness and
the GSM8K test runner), and it is the riskiest code in the project -
answer extraction is full of edge cases. Isolating it into its own
namespace gives:

- A single place to add new extractors (`\boxed{...}`, `Final answer:`,
  `####`, last-line fallback).
- A single place to extend matching (the numeric tolerance path catches
  `"1.55"` vs `"$1.55"` vs `"1.55 dollars"`).

## Why a thin HTTP server (not vLLM)

The `serve` module exists to prove the saved adapter is loadable and
addressable end-to-end without depending on heavy infrastructure. For
real serving, point users at vLLM or TGI - both load LoRA adapters with
a single CLI flag and provide proper batching and paged attention.

## Reproducibility surface

- Every run writes `run_config.json` next to the adapter.
- The Pydantic `ModelConfig` accepts an explicit `revision` so a
  specific HF SHA can be pinned per run.
- Pinned dependency versions in `pyproject.toml` (Torch, transformers,
  peft, datasets, bitsandbytes, accelerate) are part of the contract.
- The Dockerfile pins both the CUDA base image and the project deps,
  so a build today and a build six months from now produce the same
  environment.

## Known limitations

- No multi-GPU / DDP support; the current code targets single-GPU runs.
  Switching to `accelerate launch` would be straightforward but
  intentionally out of scope for the seed implementation.
- `serve.py` is single-request, single-threaded-per-handler. For real
  serving, defer to vLLM/TGI.
- Evaluation timing measures end-to-end latency only; it does not split
  out tokenizer vs generation time.
