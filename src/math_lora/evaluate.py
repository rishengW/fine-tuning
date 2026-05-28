"""Evaluation entry point.

Loads a base model (and optionally a LoRA adapter), generates completions
for either the curated :mod:`eval_prompts` set or the GSM8K test split,
extracts the final answer, and reports exact-match accuracy.

Two run modes:

- ``curated`` (default): the small hand-written prompt set in
  :mod:`math_lora.evaluation.eval_prompts`. Fast smoke check.
- ``gsm8k``: the GSM8K test split. ``--limit`` controls sample count.

Typical workflow::

    # baseline (no adapter)
    python -m math_lora.evaluate \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --suite gsm8k --limit 200 \
        --report-out reports/baseline.json

    # fine-tuned (with adapter)
    python -m math_lora.evaluate \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/adapter \
        --suite gsm8k --limit 200 \
        --report-out reports/finetuned.json

The two reports are diffable; :mod:`scripts/diff_reports.py` (or the
``make eval-diff`` target) prints the lift.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_lora.evaluation import EVAL_PROMPTS, EvalPrompt, extract_final_answer, is_match
from math_lora.evaluation.gsm8k import GSM8KExample, load_gsm8k_test
from math_lora.logging_utils import get_logger

log = get_logger("math_lora.evaluate")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run math-lora evaluation.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None, help="Path to a saved LoRA adapter.")
    p.add_argument("--suite", choices=["curated", "gsm8k"], default="curated")
    p.add_argument("--limit", type=int, default=None, help="Cap GSM8K samples.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--device", default=None, help="Override device (default: auto).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(base_model: str, adapter: str | None, device: str | None):
    log.info("loading tokenizer + base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel  # imported lazily so baseline runs without peft

        log.info("attaching LoRA adapter: %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)

    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(chosen)
    model.eval()
    return tokenizer, model, chosen


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@dataclass
class EvalRecord:
    id: str
    category: str
    question: str
    expected_answer: str
    response: str
    extracted_answer: str
    correct: bool


def generate(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
def _curated_records(prompts: tuple[EvalPrompt, ...]) -> list[tuple[str, str, str, tuple[str, ...], str]]:
    return [
        (p.id, p.category, p.question, p.answer_aliases, p.expected_answer) for p in prompts
    ]


def _gsm8k_records(examples: list[GSM8KExample]) -> list[tuple[str, str, str, tuple[str, ...], str]]:
    return [(e.id, "gsm8k", e.question, (), e.expected_answer) for e in examples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    tokenizer, model, device = load_model(args.base_model, args.adapter, args.device)

    if args.suite == "curated":
        rows = _curated_records(EVAL_PROMPTS)
    else:
        log.info("loading GSM8K test split (limit=%s)", args.limit)
        rows = _gsm8k_records(load_gsm8k_test(limit=args.limit))

    log.info("running %d prompts", len(rows))
    started = time.time()
    records: list[EvalRecord] = []
    for ex_id, category, question, aliases, expected in rows:
        response = generate(
            tokenizer,
            model,
            question,
            args.max_new_tokens,
            args.temperature,
            device,
        )
        extracted = extract_final_answer(response)
        correct = is_match(extracted, expected, aliases)
        records.append(
            EvalRecord(
                id=ex_id,
                category=category,
                question=question,
                expected_answer=expected,
                response=response,
                extracted_answer=extracted,
                correct=correct,
            )
        )

    elapsed = time.time() - started
    correct = sum(1 for r in records if r.correct)
    accuracy = correct / len(records) if records else 0.0

    log.info(
        "suite=%s n=%d correct=%d accuracy=%.4f elapsed=%.1fs",
        args.suite,
        len(records),
        correct,
        accuracy,
        elapsed,
    )

    summary: dict[str, Any] = {
        "suite": args.suite,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "n": len(records),
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "records": [asdict(r) for r in records],
    }

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info("report written: %s", args.report_out)


if __name__ == "__main__":
    main()
