# Copyright (c) 2026 BAAI. All rights reserved.
"""Emit standalone per-layer KV cache buffers instead of one packed pool.

vLLM's packed planner gives every layer a page-strided window of one shared
pool; on LPU any strict sub-interval access of such a window is staged as the
whole backing, so a single insert costs a pool-sized round trip. This patch
rebinds the planner to keep the block count and total bytes identical and
change only each tensor's shape: one independent buffer per (page_size, slot)
-- upstream's own unpacked form. The rewrite is all-or-nothing.

"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Name used in VLLM_FL_LPU_DISABLE_PATCHES.
PATCH_NAME = "kv_config_unpacked"

# A/B escape hatch: 1 restores the upstream packed planner.
_ENV_PACKED = "VLLM_FL_KV_CONFIG_PACKED"

# Idempotency marker on vllm.v1.core.kv_cache_utils.
_INSTALLED_ATTR = "_vllm_fl_kv_config_unpacked"


def _user_disabled() -> bool:
    raw = os.environ.get("VLLM_FL_LPU_DISABLE_PATCHES", "")
    return PATCH_NAME in {name.strip() for name in raw.split(",") if name.strip()}


def _packed_requested() -> bool:
    return os.environ.get(_ENV_PACKED, "0") == "1"


def plan_unpacked(
    kcu: Any,
    orig: Any,
    vllm_config: Any,
    kv_cache_groups: Any,
    available_memory: int,
):
    """``orig``'s plan with each layer given its own buffer.

    ``orig`` is called first so the block count, the reservation policy
    (``may_override_num_blocks``) and — crucially — whether upstream packs at
    all stay upstream's decision.  Only the per-tensor shape is rewritten, and
    the rewrite is validated against the packed plan before it is returned.
    """
    num_blocks, packed = orig(vllm_config, kv_cache_groups, available_memory)
    if not any(t.block_stride > 0 for t in packed):
        return num_blocks, packed  # upstream did not pack; nothing to undo

    # Ground truth for the per-slot page size: the same helper upstream used to
    # build the packed plan, so the two cannot drift apart silently.
    buckets = kcu._bucket_layers_by_page_size(kv_cache_groups)
    tensors = [
        kcu.KVCacheTensor(size=ps * num_blocks, shared_by=list(slot))
        for ps, slots in buckets.items()
        for slot in slots
    ]

    # Order-insensitive: the buffer order in the plan only decides allocation
    # order, while the layer-to-buffer association (``shared_by``) is what the
    # worker keys on.  An upstream reordering must not fail the run; a mapping
    # change must.
    def _mapping(plan):
        return sorted(tuple(t.shared_by) for t in plan)

    if _mapping(tensors) != _mapping(packed):
        raise RuntimeError(
            "LPU unpacked KV config: layer-to-buffer mapping diverged from the "
            f"packed plan (unpacked={_mapping(tensors)}, "
            f"packed={_mapping(packed)})"
        )
    strides = {t.block_stride for t in packed}
    if len(strides) != 1:
        # Not a shape this patch understands; refusing beats guessing at which
        # stride is the block size.
        raise RuntimeError(
            "LPU unpacked KV config: packed plan has inconsistent "
            f"block_stride values {sorted(strides)}"
        )
    packed_total = strides.pop() * num_blocks
    unpacked_total = sum(t.size for t in tensors)
    if unpacked_total != packed_total:
        raise RuntimeError(
            "LPU unpacked KV config: total bytes changed "
            f"({unpacked_total} != {packed_total})"
        )
    logger.info(
        "[kv-config] KV cache layout: %d buffer(s), %d blocks, %.2f GiB total "
        "(packed pool -> per-(page_size, slot) buffers)",
        len(tensors),
        num_blocks,
        unpacked_total / 2**30,
    )
    return num_blocks, tensors


def apply_kv_config_unpacked() -> bool | None:
    """Rebind the packed KV cache planner to the unpacked one.

    Idempotent; repeated calls from any process are safe.  Returns True when
    the rebind is in place, None when it was intentionally not installed (the
    knob is off or the A/B packed arm is requested -- both log their reason),
    and False only when the target symbol is missing.  The patch registry
    treats False as a declined install, so a knob must not report that way.
    """
    try:
        from vllm.v1.core import kv_cache_utils as kcu
    except ImportError:  # pragma: no cover -- vllm without a packed planner
        return False
    if _user_disabled():
        logger.info(
            "[kv-config] unpacked KV layout disabled by "
            "VLLM_FL_LPU_DISABLE_PATCHES; keeping the upstream packed pool"
        )
        return None
    if _packed_requested():
        logger.info(
            "[kv-config] %s=1: keeping the upstream packed pool (A/B arm)",
            _ENV_PACKED,
        )
        return None
    if getattr(kcu, _INSTALLED_ATTR, False):
        return True

    orig = getattr(kcu, "_get_kv_cache_config_packed", None)
    if orig is None:
        # Upstream renamed or dropped the packed planner: log loudly and leave
        # the layout alone rather than guessing at a replacement seam.
        logger.warning(
            "[kv-config] vllm.v1.core.kv_cache_utils has no "
            "_get_kv_cache_config_packed; KV cache stays packed (expect "
            "pool-sized staging cost per page access on LPU)"
        )
        return False

    def _unpacked(vllm_config, kv_cache_groups, available_memory):
        return plan_unpacked(
            kcu, orig, vllm_config, kv_cache_groups, available_memory
        )

    kcu._get_kv_cache_config_packed = _unpacked
    # Separate module-level name: a call site that imported the DSV4 spelling
    # keeps the packed planner unless the alias is rebound too.
    kcu._get_kv_cache_config_deepseek_v4 = _unpacked
    setattr(kcu, _INSTALLED_ATTR, True)
    return True


__all__ = ["PATCH_NAME", "apply_kv_config_unpacked", "plan_unpacked"]
