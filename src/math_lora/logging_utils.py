"""Logging and experiment tracking helpers.

Two responsibilities:

1. Structured stdout logging via the stdlib ``logging`` module so that
   training output is timestamped and grep-friendly.
2. Optional Weights & Biases integration. W&B is optional so the project
   stays runnable in environments without internet access; if the user
   sets ``tracking.enabled=true`` and ``wandb`` is installed, runs are
   logged there.

Anything that calls these helpers should keep working even if W&B is
missing or login fails.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str = "math_lora") -> logging.Logger:
    """Return a configured logger. Idempotent."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=os.getenv("MATH_LORA_LOG_LEVEL", "INFO"),
            format=LOG_FORMAT,
            datefmt=LOG_DATEFMT,
            stream=sys.stdout,
        )
        _configured = True
    return logging.getLogger(name)


class WandbTracker:
    """Thin wrapper around `wandb` that no-ops when disabled or unavailable."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str | None,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        self._run = None
        self._log = get_logger("math_lora.wandb")

        if not enabled:
            return

        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            self._log.warning(
                "tracking.enabled=true but `wandb` is not installed; "
                "metrics will only be logged to stdout."
            )
            return

        try:
            self._run = wandb.init(
                project=project,
                name=run_name,
                tags=tags,
                config=config,
            )
            self._log.info("wandb run initialized: %s", self._run.name)
        except Exception as exc:  # noqa: BLE001 - tracker must never crash training
            self._log.warning("wandb init failed (%s); continuing without tracking.", exc)
            self._run = None

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("wandb log failed: %s", exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("wandb finish failed: %s", exc)
