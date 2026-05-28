"""Curated prompt set for before/after fine-tuning comparison.

The prompts mirror the training distribution:

- A few grade-school word problems in the GSM8K style (the bulk of the
  training set).
- A few short calculus prompts in the same style as ``data/val.jsonl``.
- A couple of slightly harder competition-style items to probe whether the
  adapter generalizes beyond the easy cases.

Every prompt has a known final answer. We don't grade the full solution
text (a small model's intermediate steps are noisy); instead the runner
extracts the model's stated final answer and checks it against
``expected_answer``. ``answer_aliases`` lists alternative spellings that
should also count as correct (e.g. ``"x^4 + C"`` vs ``"x**4 + C"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    category: str
    question: str
    expected_answer: str
    answer_aliases: tuple[str, ...] = field(default_factory=tuple)


EVAL_PROMPTS: tuple[EvalPrompt, ...] = (
    # ---- GSM8K-style word problems --------------------------------------
    EvalPrompt(
        id="gsm-apples",
        category="word-problem",
        question=(
            "Sara has 12 apples. She gives 3 to her brother and then buys "
            "twice as many as she has left. How many apples does she have now?"
        ),
        expected_answer="27",
    ),
    EvalPrompt(
        id="gsm-train",
        category="word-problem",
        question=(
            "A train travels 60 miles in 1.5 hours. At the same constant "
            "speed, how many miles will it travel in 4 hours?"
        ),
        expected_answer="160",
        answer_aliases=("160 miles",),
    ),
    EvalPrompt(
        id="gsm-pencils",
        category="word-problem",
        question=(
            "A pencil costs 25 cents and an eraser costs 40 cents. "
            "How much do 3 pencils and 2 erasers cost in dollars?"
        ),
        expected_answer="1.55",
        answer_aliases=("$1.55", "1.55 dollars"),
    ),
    EvalPrompt(
        id="gsm-ages",
        category="word-problem",
        question=(
            "Alice is 4 years older than Bob. In 6 years, the sum of their "
            "ages will be 40. How old is Alice now?"
        ),
        expected_answer="16",
    ),
    # ---- Short calculus (matches data/val.jsonl style) ------------------
    EvalPrompt(
        id="calc-deriv-poly",
        category="calculus",
        question="Differentiate f(x) = 3x^2 + 5x - 7.",
        expected_answer="6x + 5",
        answer_aliases=("f'(x) = 6x + 5", "6*x + 5"),
    ),
    EvalPrompt(
        id="calc-int-power",
        category="calculus",
        question="Compute the indefinite integral of 6x^2.",
        expected_answer="2x^3 + C",
        answer_aliases=("2*x^3 + C", "2x**3 + C"),
    ),
    EvalPrompt(
        id="calc-def-int",
        category="calculus",
        question="Evaluate the definite integral from 0 to 2 of 3x^2 dx.",
        expected_answer="8",
    ),
    # ---- Slightly harder algebra ----------------------------------------
    EvalPrompt(
        id="alg-quadratic",
        category="algebra",
        question="Solve for x: x^2 - 5x + 6 = 0. List all real solutions.",
        expected_answer="2, 3",
        answer_aliases=(
            "x = 2, 3",
            "x = 2 or x = 3",
            "x = 2 and x = 3",
            "{2, 3}",
            "2 and 3",
        ),
    ),
    EvalPrompt(
        id="alg-system",
        category="algebra",
        question=(
            "Solve the system: 2x + y = 7 and x - y = 2. "
            "Give x and y."
        ),
        expected_answer="x = 3, y = 1",
        answer_aliases=("(3, 1)", "x=3, y=1", "x = 3 and y = 1"),
    ),
    EvalPrompt(
        id="num-percent",
        category="word-problem",
        question="What is 15% of 240?",
        expected_answer="36",
    ),
)
