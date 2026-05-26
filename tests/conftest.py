"""Top-level pytest fixtures and Hypothesis profile registration.

Hypothesis profiles
-------------------

The design document requires every property-based test to run a minimum of 100
iterations. We register two profiles here so individual tests do not need to
override ``max_examples`` themselves:

- ``default``: ``max_examples=100`` - used for local development and the standard
  ``pytest`` invocation. Satisfies the design's ``max_examples >= 100`` rule.
- ``ci``: ``max_examples=200`` - used for continuous integration where extra
  coverage is affordable. Activated by setting the environment variable
  ``HYPOTHESIS_PROFILE=ci`` before invoking pytest.

Profile selection precedence:
1. The ``HYPOTHESIS_PROFILE`` environment variable, if set.
2. Otherwise the ``default`` profile.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, Phase, Verbosity, settings


# ---------------------------------------------------------------------------
# Hypothesis profile registration
# ---------------------------------------------------------------------------

# Default profile: minimum coverage required by the design (>= 100 examples).
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,  # property tests over simulated runs can be slow on Windows
    verbosity=Verbosity.normal,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
    phases=(
        Phase.explicit,
        Phase.reuse,
        Phase.generate,
        Phase.target,
        Phase.shrink,
    ),
)

# CI profile: more thorough exploration when wall-clock budget allows.
settings.register_profile(
    "ci",
    parent=settings.get_profile("default"),
    max_examples=200,
    print_blob=True,
)

# Dev profile: tiny, fast feedback loop while iterating on a property locally.
# Not used by default; opt in with HYPOTHESIS_PROFILE=dev.
settings.register_profile(
    "dev",
    parent=settings.get_profile("default"),
    max_examples=25,
)


def _select_profile() -> str:
    """Return the Hypothesis profile name to load for this test run."""
    name = os.environ.get("HYPOTHESIS_PROFILE", "").strip().lower()
    if name in {"default", "ci", "dev"}:
        return name
    return "default"


settings.load_profile(_select_profile())
