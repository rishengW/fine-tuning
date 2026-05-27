"""Evaluate the base model and/or a LoRA adapter on a fixed prompt set.

Usage examples (from the project root, with the venv activated):

    # Run the base model only
    python -m math_lora.evaluate \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --output results/before.json

    # Run the base model + a trained adapter
    python -m math_lora.evaluate \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/adapter \
        --output results/after.json

The prompt set lives in ``math_lora.eval_prompts``. Each prompt has a known
final answer; the script extracts the model's stated final answer and
compares it (case- and whitespace-normalized) against the expected string
and any aliases. Full responses are written to the output JSON for manual
review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .eval_prompts import EVAL_PROMPTS, EvalPrompt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    base_model: str
    adapter: Path | None
    output: Path
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    seed: int
    label: str | None


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate a base model or LoRA adapter.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional path to a trained LoRA adapter directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/eval.json"),
        help="Where to write the JSON report.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling. Default is greedy decoding for reproducibility.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label",
        default=None,
        help="Free-form label stored in the report (e.g. 'before' or 'after').",
    )
    args = p.parse_args()
    return EvalConfig(
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.sample,
        seed=args.seed,
        label=args.label,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(base_model: str, adapter: Path | None):
    """Load tokenizer + model, optionally attaching a LoRA adapter."""
    dtype = pick_dtype()
    print(f"[eval] loading base model: {base_model} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if adapter is not None:
        # Imported lazily so users running base-only eval don't need peft loaded.
        from peft import PeftModel

        adapter = adapter.resolve()
        if not adapter.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")
        print(f"[eval] attaching LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Generation + answer extraction
# ---------------------------------------------------------------------------
_FINAL_ANSWER_PATTERNS = (
    re.compile(r"final\s*answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"####\s*(.+?)\s*$", re.MULTILINE),
)


def extract_final_answer(text: str) -> str:
    """Pull a single-line 'final answer' out of a free-form response.

    We try several common patterns in order: 'Final answer: ...', 'Answer: ...',
    LaTeX ``\\boxed{...}``, and the GSM8K ``#### ...`` marker. If none match,
    we fall back to the last non-empty line, which is usually a single number
    or expression for short prompts.
    """
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip().rstrip(".")
    # Fallback: last non-empty line.
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.rstrip(".")
    return ""


def normalize(s: str) -> str:
    """Loose normalization for answer comparison."""
    s = s.strip().lower()
    s = s.rstrip(".")
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s)
    # Remove a trailing period or surrounding quotes.
    s = s.strip("\"'")
    # Strip a leading "$" for currency answers.
    if s.startswith("$"):
        s = s[1:].strip()
    # Drop trailing units like "miles", "dollars" for very common cases so
    # "160" matches "160 miles". This is intentionally narrow.
    s = re.sub(
        r"\s+(miles|dollars|cents|apples|years|years old)$",
        "",
        s,
    )
    # Equalize "*" and "^" for power notation.
    s = s.replace("**", "^")
    # Equalize spacing around equals signs.
    s = re.sub(r"\s*=\s*", "=", s)
    return s


def is_correct(model_answer: str, prompt: EvalPrompt) -> bool:
    candidates = (prompt.expected_answer, *prompt.answer_aliases)
    norm_model = normalize(model_answer)
    return any(normalize(c) == norm_model for c in candidates)


def generate_one(
    tokenizer,
    model,
    device: str,
    prompt: EvalPrompt,
    cfg: EvalConfig,
) -> str:
    messages = [{"role": "user", "content": prompt.question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=cfg.temperature if cfg.temperature > 0 else 0.7,
            top_p=cfg.top_p,
        )
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    # Slice off the prompt tokens so we only decode the new completion.
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer, model, device = load_model(cfg.base_model, cfg.adapter)

    label = cfg.label or ("after" if cfg.adapter else "before")
    results: list[dict[str, Any]] = []
    correct = 0

    print(f"[eval] running {len(EVAL_PROMPTS)} prompts (label={label})")
    for i, prompt in enumerate(EVAL_PROMPTS, start=1):
        t0 = time.perf_counter()
        response = generate_one(tokenizer, model, device, prompt, cfg)
        elapsed = time.perf_counter() - t0
        extracted = extract_final_answer(response)
        ok = is_correct(extracted, prompt)
        if ok:
            correct += 1
        marker = "PASS" if ok else "FAIL"
        print(
            f"[eval] {i:>2}/{len(EVAL_PROMPTS)} [{marker}] {prompt.id} "
            f"({elapsed:.1f}s) -> {extracted!r} (expected {prompt.expected_answer!r})"
        )
        results.append(
            {
                "id": prompt.id,
                "category": prompt.category,
                "question": prompt.question,
                "expected_answer": prompt.expected_answer,
                "answer_aliases": list(prompt.answer_aliases),
                "model_response": response,
                "model_final_answer": extracted,
                "correct": ok,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    total = len(EVAL_PROMPTS)
    accuracy = correct / total if total else 0.0
    summary = {
        "label": label,
        "base_model": cfg.base_model,
        "adapter": str(cfg.adapter) if cfg.adapter else None,
        "num_prompts": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "decoding": {
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_new_tokens": cfg.max_new_tokens,
            "seed": cfg.seed,
        },
        "results": results,
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[eval] {label}: {correct}/{total} correct ({accuracy:.1%})  "
        f"-> {cfg.output}"
    )
    return summary


def main() -> None:
    cfg = parse_args()
    try:
        run_eval(cfg)
    except FileNotFoundError as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
