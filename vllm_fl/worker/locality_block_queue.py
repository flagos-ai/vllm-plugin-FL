# SPDX-License-Identifier: Apache-2.0
"""HBM Row-Buffer Locality-Preserving KV-Cache Free-Block Queue (`locality_block_queue.py`).

Intercepts `FreeKVCacheBlockQueue.append_n` / `prepend_n` to coalesce batch-freed blocks
into ascending physical `block_id` runs, preserving HBM row-buffer locality across paged KV.
"""

from __future__ import annotations

import logging
from operator import attrgetter
from typing import Any

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

_BLOCK_ID_GETTER = attrgetter("block_id")


class LocalityAwareFreeBlockQueue:
    """Physical-address coalescing hook for `vllm.v1.core.kv_cache_utils.FreeKVCacheBlockQueue`.

    As variable-length requests finish out of order, `FreeKVCacheBlockQueue`
    accumulates fragmented `block_id` sequences. Subsequent prefill/decode
    allocations then scatter a single sequence's KV pages across distant HBM
    banks/rows, degrading DRAM row-buffer hit rates and L2 spatial locality
    during paged attention Stage-1 / Decode gathers. Coalescing multi-block free
    batches into ascending `block_id` order restores contiguous physical page
    runs at negligible CPU cost.
    """

    coalesced_batches: int = 0
    coalesced_blocks: int = 0

    @staticmethod
    def coalesce_blocks(blocks: list[Any]) -> list[Any]:
        """Sort a freed block batch into ascending `block_id` order with O(n) monotonic fast paths."""
        n = len(blocks)
        if n <= 1:
            return blocks
        # Fast path 1: already ascending (common right after pool initialization).
        if blocks[0].block_id <= blocks[-1].block_id and all(
            blocks[i].block_id <= blocks[i + 1].block_id for i in range(n - 1)
        ):
            return blocks
        LocalityAwareFreeBlockQueue.coalesced_batches += 1
        LocalityAwareFreeBlockQueue.coalesced_blocks += n
        # Fast path 2: strictly descending (e.g., LIFO release) -> O(n) slice reversal.
        if blocks[0].block_id > blocks[-1].block_id and all(
            blocks[i].block_id >= blocks[i + 1].block_id for i in range(n - 1)
        ):
            return blocks[::-1]
        return sorted(blocks, key=_BLOCK_ID_GETTER)

    @classmethod
    def make_prepend_n(cls, orig_prepend_n):
        """Wrap `prepend_n` to coalesce immediately reusable uncached free blocks."""
        def _locality_aware_prepend_n(self, blocks: list[Any]) -> None:
            if len(blocks) > 1:
                blocks = cls.coalesce_blocks(blocks)
            return orig_prepend_n(self, blocks)

        return _locality_aware_prepend_n

    @classmethod
    def make_append_n(cls, orig_append_n):
        """Wrap `append_n` to coalesce unhashed blocks while preserving APC LRU order."""
        def _locality_aware_append_n(self, blocks: list[Any]) -> None:
            # When Automatic Prefix Caching (APC) is enabled, blocks with a valid
            # `block_hash` are appended in eviction-priority order; only coalesce
            # unhashed blocks (`block_hash is None`) so APC LRU semantics stay exact.
            if len(blocks) > 1 and all(
                getattr(b, "block_hash", None) is None for b in blocks
            ):
                blocks = cls.coalesce_blocks(blocks)
            return orig_append_n(self, blocks)

        return _locality_aware_append_n


def patch_kv_cache_utils_module(mod: Any) -> None:
    """Idempotently patch `FreeKVCacheBlockQueue` methods on `mod`."""
    cls = getattr(mod, "FreeKVCacheBlockQueue", None)
    if cls is None or getattr(cls, "_fl_locality_patched", False):
        return
    if hasattr(cls, "prepend_n"):
        cls.prepend_n = LocalityAwareFreeBlockQueue.make_prepend_n(cls.prepend_n)
    if hasattr(cls, "append_n"):
        cls.append_n = LocalityAwareFreeBlockQueue.make_append_n(cls.append_n)
    cls._fl_locality_patched = True
    logger.info("[FL-Worker] Activated LocalityAwareFreeBlockQueue (HBM physical block coalescing).")


def install_locality_free_block_queue() -> None:
    """Idempotently install `LocalityAwareFreeBlockQueue` onto `vllm.v1.core.kv_cache_utils`."""
    try:
        import vllm.v1.core.kv_cache_utils as kv_mod

        patch_kv_cache_utils_module(kv_mod)
    except Exception:
        return None


__all__ = [
    "LocalityAwareFreeBlockQueue",
    "install_locality_free_block_queue",
    "patch_kv_cache_utils_module",
]
