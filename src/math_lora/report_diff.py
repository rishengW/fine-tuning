"""Compute and print the lift between two evaluation reports.

Usage::

    python -m math_lora.report_diff reports/baseline.json reports/finetuned.json

Prints overall accuracy delta plus a per-category breakdown when both
reports cover the same prompt ids.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_category(report: dict[str, Any]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in report["records"]:
        c = counts[r["category"]]
        c[1] += 1
        if r["correct"]:
            c[0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Diff two evaluation reports.")
    p.add_argument("baseline", type=Path)
    p.add_argument("finetuned", type=Path)
    args = p.parse_args()

    a = _load(args.baseline)
    b = _load(args.finetuned)

    print(f"suite      : {a['suite']} (baseline) / {b['suite']} (finetuned)")
    print(f"base_model : {a['base_model']}")
    print(f"adapter    : {b['adapter']}")
    print(f"n examples : {a['n']} -> {b['n']}")
    print()

    delta = b["accuracy"] - a["accuracy"]
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    print(f"overall accuracy: {a['accuracy']:.4f} -> {b['accuracy']:.4f}  {arrow} {delta:+.4f}")
    print()

    cat_a = _per_category(a)
    cat_b = _per_category(b)
    cats = sorted(set(cat_a) | set(cat_b))
    print(f"{'category':<16}{'baseline':>15}{'finetuned':>15}{'delta':>10}")
    for cat in cats:
        c_ok, c_n = cat_a.get(cat, (0, 0))
        f_ok, f_n = cat_b.get(cat, (0, 0))
        c_acc = c_ok / c_n if c_n else 0.0
        f_acc = f_ok / f_n if f_n else 0.0
        d = f_acc - c_acc
        print(
            f"{cat:<16}"
            f"{c_ok:>4}/{c_n:<3} ({c_acc:.2f}) "
            f"{f_ok:>4}/{f_n:<3} ({f_acc:.2f}) "
            f"{d:>+8.3f}"
        )


if __name__ == "__main__":
    main()
