# Copyright (c) 2026 BAAI. All rights reserved.

"""Opt-in activation bridge for FlagGems' generic ATen plan cache."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

try:
    from vllm.logger import init_logger
except ImportError:  # Keep the bridge importable for compatibility tests.
    logger = logging.getLogger(__name__)
else:
    logger = init_logger(__name__)

_PLUGIN_ENV = "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE"
_FLAGGEMS_ENV = "FLAGGEMS_ATEN_PLAN_CACHE"
_REPORT_ENV = "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REPORT"
_REQUIRE_ENV = "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE"
_FALSE_VALUES = frozenset(("0", "false", "off", "no"))


def _cache_requested() -> bool:
    """Return the plugin override, or inherit FlagGems' opt-in switch."""

    value = os.getenv(_PLUGIN_ENV)
    if value is None:
        value = os.getenv(_FLAGGEMS_ENV, "0")
    return value.strip().lower() not in _FALSE_VALUES


def _activation_failed(reason: str) -> bool:
    """Make a requested benchmark optimization fail closed when required."""

    message = f"FlagGems ATen plan cache is not verified active: {reason}"
    if os.getenv(_REQUIRE_ENV, "0").strip().lower() not in _FALSE_VALUES:
        raise RuntimeError(message)
    logger.warning("%s; continuing without a cache-on guarantee", message)
    return False


def _log_active_identity(flag_gems: Any, backend: str) -> None:
    """Record the actual worker import, not a separate docker-exec import."""

    source = getattr(flag_gems, "__file__", None)
    source_hash = None
    if source is not None:
        with suppress(OSError):
            source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    logger.info(
        "Verified active FlagGems ATen plan cache through %s: source=%s sha256=%s",
        backend,
        source,
        source_hash,
    )


def get_flaggems_aten_plan_cache_stats() -> dict[str, Any] | None:
    """Return cache statistics across the public and compatibility ABIs."""

    try:
        import flag_gems
    except (ImportError, OSError):
        return None

    stats_fn = getattr(flag_gems, "aten_plan_cache_stats", None)
    if stats_fn is None:
        try:
            from flag_gems.utils.aten_plan_cache import cache_stats as stats_fn
        except (ImportError, OSError):
            return None
    try:
        stats = stats_fn()
    except Exception as error:  # Keep an optional optimization non-fatal.
        logger.debug("Unable to read FlagGems ATen plan-cache stats: %s", error)
        return None
    return dict(stats) if isinstance(stats, Mapping) else None


def apply_flaggems_aten_plan_cache() -> bool:
    """Install the FlagGems cache when the independently patched ABI exists.

    No accelerator-vendor assumption is made here.  An unpatched/older
    FlagGems wheel or an incompatible private ABI simply keeps its original
    routing path, which is important on platforms that do not rebuild the
    plugin's native extension.
    """

    if not _cache_requested():
        if os.getenv(_REQUIRE_ENV, "0").strip().lower() not in _FALSE_VALUES:
            return _activation_failed("required but disabled by environment")
        logger.info("FlagGems ATen plan cache disabled by environment")
        return False
    try:
        import flag_gems
    except (ImportError, OSError) as error:
        return _activation_failed(str(error))

    stats = get_flaggems_aten_plan_cache_stats()
    if stats is not None and stats.get("enabled") and stats.get("installed"):
        _log_active_identity(flag_gems, "already-enabled FlagGems API")
        return True

    enable_fn = getattr(flag_gems, "enable_aten_plan_cache", None)
    backend = "public API"
    if enable_fn is None:
        try:
            from flag_gems.utils import aten_plan_cache
        except (ImportError, OSError) as error:
            return _activation_failed(str(error))
        enable_fn = getattr(aten_plan_cache, "enable", None)
        backend = "module enable compatibility API"
        if enable_fn is None:
            enable_fn = getattr(aten_plan_cache, "install", None)
            backend = "legacy install compatibility API"
    if enable_fn is None:
        return _activation_failed("unsupported ABI")

    try:
        enabled = bool(enable_fn())
    except Exception as error:  # Keep an optional optimization non-fatal.
        return _activation_failed(str(error))
    stats = get_flaggems_aten_plan_cache_stats()
    # Legacy install() may only install wrappers without enabling lookups.
    # Never label that return value as cache-on without runtime state evidence.
    if (
        not enabled
        or not stats
        or not stats.get("enabled")
        or not stats.get("installed")
    ):
        return _activation_failed(f"{backend} returned {enabled}, stats={stats}")
    _log_active_identity(flag_gems, backend)
    return True


def log_flaggems_aten_plan_cache_stats(stage: str) -> None:
    """Log aggregate runtime evidence once at a lifecycle boundary."""

    if not _cache_requested():
        return
    stats = get_flaggems_aten_plan_cache_stats()
    if stats is None:
        _activation_failed(f"statistics unavailable at {stage}")
        return
    if not stats.get("enabled") or not stats.get("installed"):
        _activation_failed(f"inactive state at {stage}: {stats}")
    if stage == "post-warmup" and (
        stats.get("hits", 0) <= 0 or stats.get("misses", 0) <= 0
    ):
        _activation_failed(f"no demonstrated cache reuse at {stage}: {stats}")
    log_fn = logger.warning if os.getenv(_REPORT_ENV, "0") == "1" else logger.info
    log_fn(
        "FlagGems ATen plan cache stats (%s): enabled=%s installed=%s "
        "hits=%s misses=%s hit_rate=%.4f size=%s evictions=%s bypasses=%s",
        stage,
        stats.get("enabled"),
        stats.get("installed"),
        stats.get("hits", 0),
        stats.get("misses", 0),
        float(stats.get("hit_rate", 0.0)),
        stats.get("size", 0),
        stats.get("evictions", 0),
        stats.get("bypasses", 0),
    )


__all__ = [
    "apply_flaggems_aten_plan_cache",
    "get_flaggems_aten_plan_cache_stats",
    "log_flaggems_aten_plan_cache_stats",
]
