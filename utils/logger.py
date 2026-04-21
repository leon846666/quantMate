"""Unified logger.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")

Log level + file rotation config come from `config/settings.yaml`.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_once() -> None:
    """Idempotent: configure root logger on first call only."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Lazy import to avoid circular dependency with utils.config
    from utils.config import get_settings

    settings = get_settings()
    log_cfg = settings.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = Path(log_cfg.get("file", "logs/quantmate.log"))
    rotate_mb = int(log_cfg.get("rotate_mb", 20))
    keep = int(log_cfg.get("keep", 5))

    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any pre-existing handlers (e.g., from notebooks)
    root.handlers.clear()

    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    root.addHandler(sh)

    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=rotate_mb * 1024 * 1024,
        backupCount=keep,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)

    # Silence a few noisy libraries
    for noisy in ("matplotlib", "urllib3", "tushare", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name or "quantmate")
