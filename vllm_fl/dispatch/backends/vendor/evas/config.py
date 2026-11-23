# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas (LPU) centralized environment-variable configuration.

The patch-disable list lives here rather than as a raw ``os.environ.get`` call
scattered through the backend. The authoritative list of what else is tunable
lives in ``patches/registry.py``.

Usage::

    cfg = EvasConfig()
    if cfg.disable_patches:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvasConfig:
    """Centralized evas/LPU runtime configuration.

    All fields map to ``VLLM_FL_*`` environment variables.
    """

    # ---- Patch toggles ----
    # Comma-separated list of patch names to disable, e.g. "gdn,gemm"
    disable_patches: set[str] = field(
        default_factory=lambda: set(
            p.strip()
            for p in (
                os.environ.get("VLLM_FL_LPU_DISABLE_PATCHES", "")
                or os.environ.get("VLLM_FL_LPU_DISABLE_PATCH", "")
            ).split(",")
            if p.strip()
        ),
    )


# Singleton. Each invocation reads fresh from os.environ so env vars set after
# import are respected.
_evas_config: Optional[EvasConfig] = None


def get_evas_config() -> EvasConfig:
    """Return the (lazy-singleton) EvasConfig."""
    global _evas_config
    if _evas_config is None:
        _evas_config = EvasConfig()
    return _evas_config
