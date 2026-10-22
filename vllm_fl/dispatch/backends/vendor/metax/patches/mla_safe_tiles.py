# Copyright (c) 2026 BAAI. All rights reserved.

"""Fit FlagGems' sparse MLA prefill kernel into MetaX C550 shared memory.

``triton_flash_mla_sparse_fwd`` ships autotune configs tuned for Hopper
(BK=64, BH=64, num_stages 2/4 -> 188 KB). MetaX C550 exposes only 64 KB of
shared memory per workgroup, so every candidate raises
``OutOfResources: Required: 188416, Hardware limit: 65536`` and the engine dies
on the first prefill.

Replace the candidate list with the largest tile measured to fit (BK=16,
BH=32, stages=2 -> 54 KB on C550; smaller BK/BH fail Triton's dot/MMA shape
constraints). Set VLLM_FL_METAX_SAFE_MLA_TILES=0 to keep the stock configs.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SAFE_TILES = [(16, 32, 4, 2)]  # (BK, BH, num_warps, num_stages)


def _enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_SAFE_MLA_TILES", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _rewrite(module, kernel_name: str = "triton_flash_mla_sparse_fwd") -> bool:
    kernel = getattr(module, kernel_name, None)
    configs = getattr(kernel, "configs", None)
    if not configs:
        return False

    import triton

    kernel.configs = [
        triton.Config({"BK": bk, "BH": bh}, num_warps=warps, num_stages=stages)
        for bk, bh, warps, stages in _SAFE_TILES
    ]
    # Autotune results are keyed per shape; drop any computed with the
    # oversized configs (they would have raised instead of caching, but be
    # explicit so a stale entry can never resurface).
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        cache.clear()
    return True


def _apply() -> None:
    if not _enabled():
        return
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major >= 9:
        # Hopper+ has enough shared memory for the tuned tiles.
        return

    targets = []
    try:
        from flaggems_vllm.ops import flashmla_sparse as fv_sparse

        targets.append(("flaggems_vllm", fv_sparse))
    except Exception as exc:  # noqa: BLE001
        logger.debug("metax-mla-tiles: flaggems_vllm unavailable (%s)", exc)
    try:
        from flag_gems.fused import flashmla_sparse as fg_sparse

        targets.append(("flag_gems", fg_sparse))
    except Exception as exc:  # noqa: BLE001
        logger.debug("metax-mla-tiles: flag_gems unavailable (%s)", exc)

    for name, module in targets:
        try:
            if _rewrite(module):
                logger.warning(
                    "metax-mla-tiles: %s sparse MLA prefill autotune configs "
                    "rewritten to %s (shared-memory limit %s)",
                    name,
                    _SAFE_TILES,
                    "64 KB" if capability is not None else "unknown",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("metax-mla-tiles: failed to rewrite %s (%s)", name, exc)


_apply()
