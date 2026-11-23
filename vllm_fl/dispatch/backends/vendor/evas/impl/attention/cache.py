# Copyright (c) 2026 BAAI. All rights reserved.
"""Shared pure-torch paged KV slot read/write helpers for the DSV4 and GLM MLA backends.

vLLM- and LPU-free: low-level paged-cache slot indexing helpers
(``_read_slot`` / ``_write_slot`` etc.) used by both MLA backends to access
individual slots in the vLLM paged KV cache.
"""
from __future__ import annotations

import torch

__all__ = ["_read_slot", "_write_slot", "_read_contiguous_run",
           "_write_paged_runs"]


def _read_slot(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    r: int,
    s: int,
    storage_block_size: int,
    flattened: bool,
) -> None:
    """Return the cache entry for slot ``s`` of request ``r``.

    Two layouts: Direct (legacy / CPU tests, ``[num_blocks, 1,
    storage_block_size, head_dim]``, slot at ``cache[block, 0, offset]``) and
    Flattened (vLLM runtime: each block split into size-1 kernel blocks, slot
    at ``cache[block * storage_block_size + offset, 0, 0]``).
    """
    manager_idx = s // storage_block_size
    offset = s % storage_block_size
    blk = int(block_table[r, manager_idx].item())
    if flattened:
        return cache[blk * storage_block_size + offset, 0, 0]
    return cache[blk, 0, offset]


def _read_contiguous_run(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    r: int,
    s_start: int,
    s_end: int,
    storage_block_size: int,
    flattened: bool,
) -> torch.Tensor:
    """Return a slice covering slots ``[s_start, s_end)`` of request ``r``.

    The run must not cross a physical block boundary. A read (RHS slice), so
    it does not trigger LPU ``clone_split`` write errors.
    """
    manager_idx = s_start // storage_block_size
    offset_start = s_start % storage_block_size
    run_len = s_end - s_start
    blk = int(block_table[r, manager_idx].item())
    if flattened:
        flat_start = blk * storage_block_size + offset_start
        return cache[flat_start:flat_start + run_len, 0, 0]
    return cache[blk, 0, offset_start:offset_start + run_len]


def _write_slot(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    r: int,
    s: int,
    storage_block_size: int,
    flattened: bool,
    src: torch.Tensor,
) -> None:
    """Write ``src`` into slot ``s`` of request ``r``.

    See ``_read_slot`` for the two supported layouts.
    """
    manager_idx = s // storage_block_size
    offset = s % storage_block_size
    blk = int(block_table[r, manager_idx].item())
    if flattened:
        cache[blk * storage_block_size + offset, 0, 0, :].copy_(src)
    else:
        cache[blk, 0, offset, :].copy_(src)


def _write_paged_runs(
    cache: torch.Tensor,
    latent: torch.Tensor,
    slot_mapping: torch.Tensor,
    storage_block_size: int,
) -> None:
    """Write ``latent`` rows into paged-cache slots via run-grouped consecutive
    ``copy_`` (shared by GLM ``_glm_paged_write`` and DSV3 ``do_kv_cache_update``).

    Consecutive ``slot_mapping`` entries coalesce into one slice copy, capped at
    the physical block boundary; negative (padded) slots are skipped. Supports
    both layouts via Ellipsis indexing: 3D ``[num_blocks, block_size, head_size]``
    (DSV3 / upstream MLACommon) and 4D ``[num_blocks, 1, block_size, head_size]``
    (GLM). ``slot_mapping`` is the worker's device buffer (possibly -1 padded for
    graph alignment) — flattened and cloned to CPU for the timing-safe loop.
    """
    slot = slot_mapping.flatten().cpu()
    n = min(latent.shape[0], slot.numel())
    i = 0
    while i < n:
        raw = int(slot[i].item())
        if raw < 0:
            i += 1
            continue
        blk = raw // storage_block_size
        offset_start = raw % storage_block_size

        # Cap the run at the physical block boundary so the slice stays within
        # ``cache[blk, ...]`` bounds.
        max_in_block = min(
            (blk + 1) * storage_block_size - raw,
            n - i,
        )
        j = i + 1
        while j < i + max_in_block:
            val = int(slot[j].item())
            if val < 0 or val != raw + (j - i):
                break
            j += 1
        run_len = j - i

        if (blk >= cache.shape[0]
                or offset_start + run_len > storage_block_size):
            # Out of bounds (should not happen with a valid slot mapping);
            # skip this run rather than corrupting memory.
            import sys as _wsys
            _wsys.stderr.write(
                f"[PAGED_WRITE_ERR] OOB! slot={raw} blk={blk} "
                f"offset={offset_start} run_len={run_len} "
                f"cache_shape={tuple(cache.shape)}\n")
            _wsys.stderr.flush()
            i = j
            continue
        cache[blk, ..., offset_start:offset_start + run_len, :].copy_(
            latent[i:j]
        )
        i = j
