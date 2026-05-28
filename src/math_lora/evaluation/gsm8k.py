"""GSM8K test-split loader for end-to-end accuracy measurement.

GSM8K stores final answers after a ``####`` marker in the ``answer``
column. We strip that to get a clean numeric ground truth, then pair it
with the question for the harness to score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GSM8KExample:
    id: str
    question: str
    expected_answer: str


_HASH_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")


def load_gsm8k_test(limit: int | None = None) -> list[GSM8KExample]:
    """Load the GSM8K test split as a list of :class:`GSM8KExample`.

    Parameters
    ----------
    limit:
        If set, return only the first ``limit`` examples. Useful for smoke
        runs and CI. ``None`` returns the full ~1.3k-example split.
    """

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out: list[GSM8KExample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        match = _HASH_ANSWER_RE.search(row["answer"] or "")
        if match is None:
            continue
        # GSM8K answers are strings like "72,000" or "5"; normalize commas.
        answer = match.group(1).replace(",", "").strip()
        out.append(
            GSM8KExample(
                id=f"gsm8k-test-{i:04d}",
                question=row["question"].strip(),
                expected_answer=answer,
            )
        )
    return out
