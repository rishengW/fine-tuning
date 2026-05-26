"""Placeholder so the ``tests/integration`` directory is discoverable by pytest.

Real integration tests (real GSM8K/MATH scoring on a small sample, an
end-to-end training run on a tiny model, an adapter-merge round-trip, a
cloud-cost reconciliation, and a reproducibility check) are added in tasks
9.x, 10.x, 11.x, 12.x.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_integration_layout_is_discoverable() -> None:
    assert True
