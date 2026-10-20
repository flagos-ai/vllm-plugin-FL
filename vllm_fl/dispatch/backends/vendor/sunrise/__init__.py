# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise vendor backend for vllm-plugin-FL dispatch."""

import logging as _logging
import os as _os

_log = _logging.getLogger(__name__)

# Default SSM conv state layout for upstream Triton causal_conv1d.
if not _os.environ.get("VLLM_SSM_CONV_STATE_LAYOUT"):
    _os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
    _log.info(
        "Sunrise vendor: defaulting VLLM_SSM_CONV_STATE_LAYOUT=DS "
        "(upstream Triton causal_conv1d)."
    )


_SAFE_FLASH_HEURISTIC_NAMES = frozenset(
    {
        "mha_block_128",
        "mha_block_64",
        "mha_block_32",
        "mha_block_16",
        "mha_varlen_decode",
    }
)


def _force_safe_flash_attn_launch_config() -> bool:
    """Pin FlagGems paged-flash heuristics to a TP-invariant safe tile.

    Default heuristic bins (esp. ``mha_block_128`` with ``BLOCK_N=8`` /
    ``mha_block_64`` with ``num_stages=3``) still produce trailing-tile NaNs
    under some per-rank ``(h, hk)`` shapes even with ``OFF_ASYNC=1``. Forcing
    the ``mha_block_16`` launch shape makes TP=2/TP=4 share one known-good
    specialization. ``num_stages`` is part of Triton's cache key.
    """
    try:
        from flag_gems import runtime as _gems_runtime
        from flag_gems.runtime.backend._sunrise.heuristics_config_utils import (
            HEURISTICS_CONFIGS,
        )
    except Exception as exc:  # pragma: no cover
        _log.debug("Skipping safe flash-tile clamp: %s", exc)
        return False

    overrides = {
        "BLOCK_M": lambda args: 16,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 8,
        "num_stages": lambda args: 1,
    }

    patched = False
    for name in _SAFE_FLASH_HEURISTIC_NAMES:
        cfg = HEURISTICS_CONFIGS.get(name)
        if isinstance(cfg, dict):
            cfg.update(overrides)
            patched = True

    if not getattr(_gems_runtime, "_sunrise_safe_flash_tile_wrapped", False):
        _orig_get = _gems_runtime.get_heuristic_config

        def _get_heuristic_config_safe(op_name):
            cfg = _orig_get(op_name)
            if op_name in _SAFE_FLASH_HEURISTIC_NAMES and isinstance(cfg, dict):
                cfg = dict(cfg)
                cfg.update(overrides)
            return cfg

        _gems_runtime.get_heuristic_config = _get_heuristic_config_safe
        _gems_runtime._sunrise_safe_flash_tile_wrapped = True
        patched = True

    if patched:
        _log.info(
            "Sunrise vendor: forced FlagGems mha_block_* launch config to "
            "BLOCK=16/16 warps=8 stages=1 (TP-invariant safe flash tile)."
        )
    return patched


_force_safe_flash_attn_launch_config()

# Import-time patches (FLA/GDN rebinds, pointwise, INT8, profiler, etc.).
from . import patches  # noqa: F401, E402

from .sunrise import SunriseBackend  # noqa: E402

__all__ = ["SunriseBackend"]
