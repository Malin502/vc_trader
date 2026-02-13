"""Structured logging with loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure loguru sinks (call once at startup)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default sink
    logger.remove()

    # Console
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )

    # File – rotate daily, keep 30 days
    logger.add(
        str(log_path / "aivc_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
    )

    _CONFIGURED = True


def get_logger(name: str = "aivc"):
    """Return a contextualized logger."""
    setup_logger()
    return logger.bind(context=name)
