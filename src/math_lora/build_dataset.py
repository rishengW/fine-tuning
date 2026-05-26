"""Build a balanced math-reasoning training set from public Hugging Face datasets.

Combines a clean grade-school source (GSM8K) with a slice of competition-style
data (NuminaMath-CoT), filters by length to keep examples within the small
base model's context window, and writes shuffled train/val jsonl files in the
same chat format expected by ``train.py``.

Why this shape:

- A 0.5B base model has limited capacity. Overly long olympiad solutions and
  noisy traces hurt more than they help, so we cap response length and skew
  the mix toward the cleaner GSM8K source.
- LoRA tends to overfit on tiny corpora; a few thousand diverse examples
  works better than tens of examples repeated for many epochs.

Usage:

    python -m math_lora.build_dataset \
        --train-out data/train.jsonl \
        --val-out data/val.jsonl \
        --train-size 3000 \
        --val-size 300
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BuildConfig:
    train_out: Path
    val_out: Path
    train_size: int
    val_size: int
    gsm8k_ratio: float
    max_chars: int
    seed: int


def parse_args() -> BuildConfig:
    p = argparse.ArgumentParser(description="Build a math reasoning JSONL dataset.")
    p.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--val-out", type=Path, default=Path("data/val.jsonl"))
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument(
        "--gsm8k-ratio",
        type=float,
        default=0.7,
        help="Fraction of training data drawn from GSM8K (rest from NuminaMath-CoT).",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Drop records whose total prompt+response length exceeds this many chars.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return BuildConfig(**vars(args))


# ---------------------------------------------------------------------------
# Source: GSM8K
# ---------------------------------------------------------------------------
_GSM8K_HASH_ANSWER = re.compile(r"####\s*(.+)\s*$")


def gsm8k_to_messages(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a GSM8K row into chat-format messages.

    GSM8K answers look like::

        "Step 1 ...
         Step 2 ...
         #### 42"

    We rewrite the trailing ``#### 42`` marker as ``Final answer: 42`` so
    the response shape matches our seed examples.
    """
    question = (record.get("question") or "").strip()
    answer = (record.get("answer") or "").strip()
    if not question or not answer:
        return None

    match = _GSM8K_HASH_ANSWER.search(answer)
    if match is None:
        return None
    final = match.group(1).strip()
    body = _GSM8K_HASH_ANSWER.sub("", answer).strip()
    response = f"{body}\nFinal answer: {final}"
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
    }


def iter_gsm8k() -> Iterator[dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for row in ds:
        out = gsm8k_to_messages(row)
        if out is not None:
            yield out


# ---------------------------------------------------------------------------
# Source: NuminaMath-CoT
# ---------------------------------------------------------------------------
def numina_to_messages(record: dict[str, Any]) -> dict[str, Any] | None:
    problem = (record.get("problem") or "").strip()
    solution = (record.get("solution") or "").strip()
    if not problem or not solution:
        return None
    # NuminaMath solutions usually contain a boxed final answer; we leave the
    # solution as-is rather than reformatting, so the model learns both styles.
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
    }


def iter_numina() -> Iterator[dict[str, Any]]:
    # Streaming avoids downloading the full ~860k-row dataset to disk.
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    for row in ds:
        out = numina_to_messages(row)
        if out is not None:
            yield out


# ---------------------------------------------------------------------------
# Filtering and assembly
# ---------------------------------------------------------------------------
def record_chars(record: dict[str, Any]) -> int:
    return sum(len(m["content"]) for m in record["messages"])


def collect(source: Iterator[dict[str, Any]], target: int, max_chars: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in source:
        if record_chars(record) > max_chars:
            continue
        out.append(record)
        if len(out) >= target:
            break
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()
    rng = random.Random(cfg.seed)

    total = cfg.train_size + cfg.val_size
    n_gsm8k = int(round(total * cfg.gsm8k_ratio))
    n_numina = total - n_gsm8k

    # Pull a bit more than we need so the length filter has slack.
    gsm8k_target = int(n_gsm8k * 1.2) + 50
    numina_target = int(n_numina * 1.2) + 50

    print(f"[build] collecting up to {gsm8k_target} GSM8K examples")
    gsm8k_records = collect(iter_gsm8k(), gsm8k_target, cfg.max_chars)
    print(f"[build] kept {len(gsm8k_records)} GSM8K examples after length filter")

    print(f"[build] streaming up to {numina_target} NuminaMath-CoT examples")
    numina_records = collect(iter_numina(), numina_target, cfg.max_chars)
    print(f"[build] kept {len(numina_records)} NuminaMath examples after length filter")

    rng.shuffle(gsm8k_records)
    rng.shuffle(numina_records)

    pool = gsm8k_records[:n_gsm8k] + numina_records[:n_numina]
    if len(pool) < total:
        print(
            f"[build] warning: only {len(pool)} examples available, "
            f"requested {total}. Consider raising --max-chars."
        )
    rng.shuffle(pool)

    train = pool[: cfg.train_size]
    val = pool[cfg.train_size : cfg.train_size + cfg.val_size]

    write_jsonl(cfg.train_out, train)
    write_jsonl(cfg.val_out, val)
    print(f"[build] wrote {len(train)} train -> {cfg.train_out}")
    print(f"[build] wrote {len(val)} val   -> {cfg.val_out}")


if __name__ == "__main__":
    main()
