# Copyright (c) 2026 BAAI. All rights reserved.
"""Activate an available ATen plan cache using FlagGems' own setting."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def apply_flaggems_aten_plan_cache() -> bool:
    """Use the available enable API; older wheels keep their normal routing."""
    if os.getenv("FLAGGEMS_ATEN_PLAN_CACHE", "0").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False

    import flag_gems

    enable = getattr(flag_gems, "enable_aten_plan_cache", None)
    if enable is None:
        try:
            from flag_gems.utils.aten_plan_cache import enable
        except ImportError:
            logger.warning("FlagGems ATen plan cache API is unavailable")
            return False
    return bool(enable())
