"""Final-answer extraction and matching.

The model's full solution text is noisy, but the *stated* final answer is
usually structured. We extract it with three falling-through rules:

1. ``\\boxed{...}`` - the LaTeX convention used in MATH and NuminaMath
   solutions.
2. ``Final answer: ...`` - the convention enforced by our training data.
3. ``#### ...`` - the GSM8K marker, kept as a fallback.

If none match, we use the last non-empty line as a last resort. Matching
is then done by lightweight normalization (whitespace, case, surrounding
``$``) plus a numeric path that compares floats with a small tolerance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_ANSWER_RE = re.compile(r"final answer\s*[:\-]\s*(.+)$", re.IGNORECASE)
_HASH_ANSWER_RE = re.compile(r"####\s*(.+)$")


def extract_final_answer(text: str) -> str:
    """Return the model's stated final answer, or the last line if none found."""
    if not text:
        return ""

    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = _FINAL_ANSWER_RE.search(line)
        if m:
            return m.group(1).strip()
        m = _HASH_ANSWER_RE.search(line)
        if m:
            return m.group(1).strip()

    # Last resort: trailing non-empty line.
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


_PUNCT_RE = re.compile(r"[\s$\\]+")


def _normalize(s: str) -> str:
    """Whitespace, dollar-sign and case insensitive comparison key."""
    return _PUNCT_RE.sub("", s).lower().strip(".,;:")


def _try_float(s: str) -> float | None:
    """Best-effort numeric parsing that strips common units and commas."""
    cleaned = re.sub(r"[,\s$%]", "", s)
    cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)  # trailing units
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_match(
    predicted: str,
    expected: str,
    aliases: Iterable[str] = (),
    *,
    rel_tol: float = 1e-3,
) -> bool:
    """Return True if ``predicted`` matches ``expected`` (or any alias)."""
    if not predicted:
        return False

    candidates = [expected, *aliases]

    pred_norm = _normalize(predicted)
    for cand in candidates:
        if pred_norm == _normalize(cand):
            return True

    # Numeric fallback for cases like "1.55" vs "1.55 dollars" vs "$1.55".
    pred_f = _try_float(predicted)
    if pred_f is None:
        return False
    for cand in candidates:
        cand_f = _try_float(cand)
        if cand_f is None:
            continue
        denom = max(abs(cand_f), 1.0)
        if abs(pred_f - cand_f) / denom <= rel_tol:
            return True
    return False
