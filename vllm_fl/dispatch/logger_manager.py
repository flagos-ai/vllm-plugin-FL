# Copyright (c) 2026 BAAI. All rights reserved.

"""
Logger manager for vllm-plugin-FL dispatch.

Provides centralized logging configuration for the dispatch module.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Default log level from environment variable
_DEFAULT_LOG_LEVEL = os.environ.get("VLLM_FL_LOG_LEVEL", "INFO").upper()

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "vllm_fl.dispatch") -> logging.Logger:
    """
    Get a logger instance for the dispatch module.

    Args:
        name: Logger name, defaults to "vllm_fl.dispatch"

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set log level from environment
        level = getattr(logging, _DEFAULT_LOG_LEVEL, logging.INFO)
        logger.setLevel(level)

    _loggers[name] = logger
    return logger


def set_log_level(level: str, name: Optional[str] = None) -> None:
    """
    Set the log level for dispatch loggers.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Optional logger name, if None sets for all cached loggers
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if name is not None:
        if name in _loggers:
            _loggers[name].setLevel(log_level)
    else:
        for logger in _loggers.values():
            logger.setLevel(log_level)
