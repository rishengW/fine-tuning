"""Unit tests for the answer extractor and matcher.

These run without torch/transformers, so they are safe in CI on machines
without a GPU.
"""

from __future__ import annotations

from math_lora.evaluation.extract import extract_final_answer, is_match


def test_extract_boxed_wins() -> None:
    text = "Step 1: ...\nFinal answer: 42\nSo \\boxed{36}."
    assert extract_final_answer(text) == "36"


def test_extract_final_answer_marker() -> None:
    text = "Step 1: 12 / 3 = 4.\nFinal answer: 4"
    assert extract_final_answer(text) == "4"


def test_extract_hash_marker() -> None:
    text = "Working...\n#### 27"
    assert extract_final_answer(text) == "27"


def test_extract_falls_back_to_last_line() -> None:
    text = "Some prose without a marker.\n\n5"
    assert extract_final_answer(text) == "5"


def test_extract_empty() -> None:
    assert extract_final_answer("") == ""


def test_match_exact() -> None:
    assert is_match("4", "4")
    assert not is_match("5", "4")


def test_match_alias() -> None:
    assert is_match("160 miles", "160", aliases=("160 miles",))


def test_match_numeric_dollars() -> None:
    assert is_match("$1.55", "1.55")
    assert is_match("1.55 dollars", "1.55")


def test_match_numeric_tolerance() -> None:
    # 0.0009 / 1.55 < 1e-3, so within tolerance
    assert is_match("1.5509", "1.55")
    # 0.01 / 1.55 > 1e-3, so out of tolerance
    assert not is_match("1.56", "1.55")


def test_match_normalized_whitespace_and_case() -> None:
    assert is_match(" Six ", "six")
    assert is_match("F'(X) = 6X + 5", "f'(x) = 6x + 5")
