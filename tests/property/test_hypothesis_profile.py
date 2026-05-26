"""Smoke property test verifying the Hypothesis configuration from task 1.1.

The design document requires ``max_examples >= 100`` per property test. This
smoke test asserts that the active Hypothesis profile honors that contract so
later property tests can rely on it without re-configuring ``settings``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.property
def test_active_hypothesis_profile_meets_minimum_examples() -> None:
    # The active profile is loaded by ``tests/conftest.py``. The design rule is
    # ``max_examples >= 100`` per property test.
    assert settings().max_examples >= 100


@pytest.mark.property
@given(st.integers(min_value=-1_000_000, max_value=1_000_000))
def test_integer_negation_is_self_inverse(n: int) -> None:
    # Trivial property used to confirm the runner picks up @given decorators.
    assert -(-n) == n
