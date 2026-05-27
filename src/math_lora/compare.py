"""Run before/after evaluations and print a side-by-side comparison.

Usage:

    python -m math_lora.compare --adapter outputs/adapter

This runs the base model first, then the base model + adapter, then prints
per-prompt and aggregate accuracy, plus the deltas.

If you already produced JSON reports with ``math_lora.evaluate``, you can
skip the model runs and just diff them:

    python -m math_lora.compare \
        --before-report results/before.json \
        --after-report  results/after.json \
        --report-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import EvalConfig, run_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base model vs LoRA adapter.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--adapter",
        type=Path,
        default=Path("outputs/adapter"),
        help="Path to the trained LoRA adapter directory.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where to write before.json / after.json.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling. Default is greedy.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Skip model runs and just compare two existing JSON reports.",
    )
    p.add_argument(
        "--before-report",
        type=Path,
        default=None,
        help="Existing JSON report to use as 'before' (with --report-only).",
    )
    p.add_argument(
        "--after-report",
        type=Path,
        default=None,
        help="Existing JSON report to use as 'after' (with --report-only).",
    )
    return p.parse_args()


def make_eval_cfg(
    base_model: str,
    adapter: Path | None,
    output: Path,
    max_new_tokens: int,
    sample: bool,
    seed: int,
    label: str,
) -> EvalConfig:
    return EvalConfig(
        base_model=base_model,
        adapter=adapter,
        output=output,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=sample,
        seed=seed,
        label=label,
    )


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_table(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_by_id = {r["id"]: r for r in before["results"]}
    after_by_id = {r["id"]: r for r in after["results"]}
    ids = list(before_by_id.keys())

    id_w = max(len("prompt"), max(len(i) for i in ids))
    cols = (
        ("prompt", id_w),
        ("expected", 14),
        ("before", 18),
        ("after", 18),
        ("Δ", 6),
    )
    header = " | ".join(name.ljust(w) for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print()
    print(header)
    print(sep)

    flips_better = 0
    flips_worse = 0
    for pid in ids:
        b = before_by_id[pid]
        a = after_by_id.get(pid)
        if a is None:
            continue
        b_ok = b["correct"]
        a_ok = a["correct"]
        if a_ok and not b_ok:
            delta = "+1"
            flips_better += 1
        elif b_ok and not a_ok:
            delta = "-1"
            flips_worse += 1
        else:
            delta = "="
        row = (
            pid.ljust(id_w),
            truncate(b["expected_answer"], 14).ljust(14),
            truncate(b["model_final_answer"], 18).ljust(18),
            truncate(a["model_final_answer"], 18).ljust(18),
            delta.ljust(6),
        )
        print(" | ".join(row))
    print(sep)

    b_acc = before["accuracy"]
    a_acc = after["accuracy"]
    print(
        f"before: {before['num_correct']}/{before['num_prompts']} "
        f"({b_acc:.1%})    "
        f"after: {after['num_correct']}/{after['num_prompts']} "
        f"({a_acc:.1%})    "
        f"delta: {a_acc - b_acc:+.1%}"
    )
    print(f"flips: +{flips_better} better, -{flips_worse} worse")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.report_only:
        if args.before_report is None or args.after_report is None:
            raise SystemExit(
                "--report-only requires --before-report and --after-report"
            )
        before = load_report(args.before_report)
        after = load_report(args.after_report)
    else:
        results_dir = args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        before_path = results_dir / "before.json"
        after_path = results_dir / "after.json"

        print("[compare] === running BEFORE (base model only) ===")
        before = run_eval(
            make_eval_cfg(
                args.base_model,
                None,
                before_path,
                args.max_new_tokens,
                args.sample,
                args.seed,
                "before",
            )
        )

        print()
        print("[compare] === running AFTER (base + adapter) ===")
        after = run_eval(
            make_eval_cfg(
                args.base_model,
                args.adapter,
                after_path,
                args.max_new_tokens,
                args.sample,
                args.seed,
                "after",
            )
        )

    print_table(before, after)


if __name__ == "__main__":
    main()
