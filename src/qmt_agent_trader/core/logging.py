"""Logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=False)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "qmt-agent-trader.log", rotation="10 MB", retention="30 days")
