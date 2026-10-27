# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""scatter_nd_update_v2 - Ascend Triton port of ``npu_scatter_nd_update_v2``.

This is the Triton implementation of the AscendC ``scatter_nd_update_v2``
operator (the same kernel validated as L1 on 910C), wired as the Ascend vendor
implementation of the dispatch op **``deepseek_v4_cache_write``** (the DeepSeek
V4 paged-cache leaf writer; the dispatch op key is intentionally unchanged here
-- the interface rename to ``paged_cache_write`` is owned by the integration
branch so this PR does not touch ``ops.py``/``kv_cache`` callers).

Dispatch contract of the host-facing entry
------------------------------------------
::

    deepseek_v4_cache_write(values, slots, cache) -> None

    values : [tokens, dim]        token values (bf16 / fp16 / fp32 / int8)
    slots  : [tokens]             linear paged slot ids; negative == padding
    cache  : [pages, block, dim]  paged cache, a view into the real vLLM packed
                                  KV-cache store: non-contiguous page strides,
                                  per-channel strides and arbitrary
                                  ``storage_offset`` are all supported in place

The cache is mutated in place at the valid slots; negative and out-of-range
slots are skipped and never alias the slot-zero row.  The only caller
(``vllm_fl.kv_cache.deepseek_v4.DeepseekV4CacheLayer.store``) guarantees that
valid slots are unique (the scheduler plan allocates each output slot at most
once); padding rows carry ``-1``.

Implementation
--------------
``slots`` is lifted to the K=2 scatter-index form::

    indices = torch.stack((slots // block, slots % block), dim=-1)

and ``cache`` itself is the in-place scatter destination.  The kernel addresses
rows through ``cache.stride()[:2]`` relative to ``cache.data_ptr()`` (which
already contains ``storage_offset``) and writes each row through
``cache.stride(2)``, so padded pages, packed-store views (element spans beyond
``2**31``) and packed last-dimension views (``cache[..., ::2]``) are all written
directly, without a cache-sized copy.  All address arithmetic is ``int64``
(including the per-channel stride term); the ``storage_numel - storage_offset``
bound is only used to reject out-of-view rows.

Unsupported layouts (zero/negative strides, an index->address map that can alias
two slots onto one row, non-integer slots, oversized values) raise
``UnsupportedLayout`` **before any device write**.  This implementation does not
catch device/launch errors and does not call the reference writer itself: the FL
dispatch manager applies its normal priority fallback (``strict: false`` ->
``reference.torch``) when a candidate raises, which keeps the vendor impl free of
hidden fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

#: Fixed grid: two programs per 910C vector core.  Deliberately not tunable via
#: env -- the tuning experiments (48/96/192) are recorded in the PR description.
_NUM_PROGRAMS = 96
_MAX_BLOCK_WIDTH = 1024
_INT32_MAX = 2**31 - 1
_SUPPORTED_CACHE_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int8,
)

#: Distinct unsupported-layout messages already logged (raise + fallback keeps
#: the hot path quiet without hiding the first occurrence).
_LOGGED_REASONS: set[str] = set()


def _log_once(reason: str, message: str) -> None:
    if reason in _LOGGED_REASONS:
        return
    _LOGGED_REASONS.add(reason)
    logger.warning(message)


class UnsupportedLayout(RuntimeError):
    """The cache layout cannot be written by this Triton implementation.

    Raised before any device write so the FL dispatch manager can fall back to
    the next registered implementation (``reference.torch``) under the normal
    ``strict: false`` policy.  There is no in-impl fallback.
    """


@triton.jit
def _cache_write_kernel(
    VALUES,  # *values_dtype [tokens, width] contiguous
    INDICES,  # *int32/int64 [tokens, 2] contiguous
    CACHE,  # *cache_dtype; base pointer == cache.data_ptr()
    TOKENS,  # runtime int
    WIDTH,  # runtime int, row width == cache.shape[-1]
    STRIDE0,  # cache.stride(0)  (int64 arithmetic below)
    STRIDE1,  # cache.stride(1)
    STRIDE2,  # cache.stride(2); per-channel stride for packed views
    ROW_SPAN,  # (width - 1) * stride(2): offset of the last row element
    PAGES,  # cache.shape[0]
    BLOCK,  # cache.shape[1]
    LIMIT,  # storage elements addressable from cache.data_ptr()
    NC: tl.constexpr,
    BLK_W: tl.constexpr,
    TAIL_CONTIGUOUS: tl.constexpr,
):
    """Scatter token rows into the paged cache (K=2, in place).

    grid=(NC,) with a runtime-stride row loop keeps the block count independent
    of the token count: on 910C a constant 48/96-program grid stays at 0.9-1.2x
    the native AscendC op in eager mode and 1.8-2.3x faster in graph replay,
    whereas grid=token_count degrades 1.2-7x as the token count grows.

    ``TAIL_CONTIGUOUS`` is the fast path (``cache.stride(2) == 1``): the row is
    written as one linear range.  Otherwise each channel is addressed through
    ``cache.stride(2)`` in int64 (packed last-dimension views such as
    ``cache[..., ::2]``); this is correct but does not vectorize as well.

    ``ROW_SPAN`` is resolved on the host instead of being computed here: Triton
    specialises scalar arguments equal to 1 (e.g. ``width=1`` or ``stride2=1``)
    into Python ints, so calling ``.to()`` on them raises at compile time.
    """
    pid = tl.program_id(0)
    for row in range(pid, TOKENS, NC):
        page = tl.load(INDICES + row * 2).to(tl.int64)
        offset = tl.load(INDICES + row * 2 + 1).to(tl.int64)
        address = page * STRIDE0 + offset * STRIDE1
        valid = (
            (page >= 0)
            & (page < PAGES)
            & (offset >= 0)
            & (offset < BLOCK)
            & (address >= 0)
            & (address + ROW_SPAN < LIMIT)
        )
        address = tl.minimum(tl.maximum(address, 0), LIMIT - 1)
        for start in range(0, tl.cdiv(WIDTH, BLK_W)):
            cols = start * BLK_W + tl.arange(0, BLK_W)
            mask = valid & (cols < WIDTH)
            source = tl.load(VALUES + row * WIDTH + cols, mask=mask, other=0)
            if TAIL_CONTIGUOUS:
                target = CACHE + address + cols
            else:
                # int64 per-channel stride: storage offset may exceed 2**31.
                target = CACHE + address + cols.to(tl.int64) * STRIDE2
            tl.store(target, source, mask=mask)


@dataclass(frozen=True)
class _WritePlan:
    """Everything the kernel launch needs, resolved on the host."""

    values: torch.Tensor
    indices: torch.Tensor
    cache: torch.Tensor
    tokens: int
    width: int
    stride0: int
    stride1: int
    stride2: int
    row_span: int
    pages: int
    block: int
    limit: int
    tail_contiguous: bool


#: Cache for the (pages, block, stride0, stride1, stride2, width) alias analysis.
_ADDRESS_MAP_SAFE_CACHE: dict[tuple[int, int, int, int, int, int], bool] = {}


def _storage_limit(cache: torch.Tensor) -> int:
    """Elements addressable from ``cache.data_ptr()`` to the storage end."""
    try:
        storage_bytes = int(cache.untyped_storage().nbytes())
        element_size = int(cache.element_size())
        storage_offset = int(cache.storage_offset())
    except Exception as exc:  # noqa: BLE001 - layout without a usable storage view
        raise UnsupportedLayout("storage_unknown") from exc
    if element_size <= 0:
        raise UnsupportedLayout("storage_unknown")
    limit = storage_bytes // element_size - storage_offset
    if limit <= 0:
        raise UnsupportedLayout("storage_unknown")
    return limit


def _address_map_is_safe(
    pages: int,
    block: int,
    stride0: int,
    stride1: int,
    stride2: int,
    width: int,
) -> bool:
    """True iff two different valid slots cannot touch overlapping rows.

    A row occupies ``row_span + 1`` storage elements starting at its slot
    address, with ``row_span = (width - 1) * stride2`` (``stride2 == 1`` is the
    compact row).  Caller-guaranteed unique slots only imply disjoint writes
    when the slot -> address map is injective at row granularity; a rare
    misaligned/overlapping layout is proven (or rejected) by looking for a
    non-zero ``page_delta * stride0 + offset_delta * stride1`` whose magnitude
    is within one row footprint.  All arithmetic is Python int (unbounded).
    """
    row_span = (width - 1) * stride2
    page_span = (block - 1) * stride1 + row_span + 1
    pages_disjoint = pages <= 1 or page_span <= stride0
    if (
        stride2 == 1
        and pages_disjoint
        and stride0 % width == 0
        and stride1 % width == 0
    ):
        return True
    key = (pages, block, stride0, stride1, stride2, width)
    cached = _ADDRESS_MAP_SAFE_CACHE.get(key)
    if cached is not None:
        return cached
    safe = True
    for page_delta in range(1 - pages, pages):
        # Nearest offset delta to -page_delta * stride0 / stride1 is the integer
        # around the floor of that quotient; +-1 covers the rounding direction.
        quotient = -(page_delta * stride0) // stride1
        for offset_delta in (quotient - 1, quotient, quotient + 1, quotient + 2):
            if abs(offset_delta) > block - 1:
                continue
            if page_delta == 0 and offset_delta == 0:
                continue
            if abs(page_delta * stride0 + offset_delta * stride1) <= row_span:
                safe = False
                break
        if not safe:
            break
    _ADDRESS_MAP_SAFE_CACHE[key] = safe
    return safe


def _build_indices(slots: torch.Tensor, block: int) -> torch.Tensor:
    """Lift 1-D linear slots to the K=2 scatter index form ``[tokens, 2]``.

    The index dtype follows ``slots`` (int32 -> int32, int64 -> int64) so an
    out-of-range int64 slot can never wrap into a valid slot through a narrower
    cast.  Negative slots keep a negative page id and are skipped in the kernel.
    """
    if slots.dtype not in (torch.int32, torch.int64):
        raise UnsupportedLayout("slots_dtype")
    return torch.stack((slots // block, slots % block), dim=-1)


def _plan(values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor) -> _WritePlan:
    """Resolve the write plan or raise ``UnsupportedLayout`` (no fallback here)."""
    pages, block, width = (int(size) for size in cache.shape)
    stride0, stride1, stride2 = (int(stride) for stride in cache.stride())
    if stride0 <= 0 or stride1 <= 0 or stride2 <= 0:
        raise UnsupportedLayout("negative_or_zero_stride")
    limit = _storage_limit(cache)
    # The kernel addresses the cache in int64, so a large packed backing tensor
    # (vLLM's per-group KV cache store, whose element span can exceed 2**31) is
    # the normal production layout and must stay on the Triton path.
    if not _address_map_is_safe(pages, block, stride0, stride1, stride2, width):
        raise UnsupportedLayout("address_map_alias")
    if values.dtype != cache.dtype:
        # The reference writer stores values in the cache dtype; keep bit parity.
        values = values.to(cache.dtype)
    if not values.is_contiguous():
        values = values.contiguous()
    if values.numel() > _INT32_MAX:
        # row * width stays int32 in the kernel; the real cache writes are tiny.
        raise UnsupportedLayout("values_too_large")
    indices = _build_indices(slots, block)
    if not indices.is_contiguous():
        indices = indices.contiguous()
    return _WritePlan(
        values=values,
        indices=indices,
        cache=cache,
        tokens=int(values.shape[0]),
        width=width,
        stride0=stride0,
        stride1=stride1,
        stride2=stride2,
        row_span=(width - 1) * stride2,
        pages=pages,
        block=block,
        limit=limit,
        tail_contiguous=stride2 == 1,
    )


def _launch(plan: _WritePlan) -> None:
    block_width = min(triton.next_power_of_2(plan.width), _MAX_BLOCK_WIDTH)
    _cache_write_kernel[(_NUM_PROGRAMS,)](
        plan.values,
        plan.indices,
        plan.cache,
        plan.tokens,
        plan.width,
        plan.stride0,
        plan.stride1,
        plan.stride2,
        plan.row_span,
        plan.pages,
        plan.block,
        plan.limit,
        NC=_NUM_PROGRAMS,
        BLK_W=block_width,
        TAIL_CONTIGUOUS=plan.tail_contiguous,
    )


def _check_contract(
    values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor
) -> None:
    """Mirror the reference provider's validation, message included."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    if cache.dtype not in _SUPPORTED_CACHE_DTYPES:
        raise ValueError("Unsupported cache storage dtype")


def scatter_nd_update_v2(
    values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor
) -> None:
    """Write ``values`` rows into the paged ``cache`` at ``slots`` in place.

    Args:
        values: ``[tokens, dim]`` token values; a non-contiguous input is copied.
        slots: ``[tokens]`` linear slot ids; negative or out-of-range ids are
            padding and are skipped.  Valid ids must be unique (caller contract).
        cache: ``[pages, block, dim]`` paged cache to mutate; non-contiguous
            page/channel strides and a storage offset are addressed in place.

    Returns:
        None.  The cache is mutated in place.

    Raises:
        ValueError: the reference provider rejects the same shapes/dtypes.
        UnsupportedLayout: the layout cannot be written safely; the FL dispatch
            manager then falls back to ``reference.torch`` (normal priority
            fallback, ``strict: false``).  No in-implementation fallback.
    """
    _check_contract(values, slots, cache)
    if values.shape[0] == 0:
        return
    if any(size == 0 for size in cache.shape):
        raise ValueError("Cannot write to an empty cache")
    try:
        plan = _plan(values, slots, cache)
    except UnsupportedLayout as exc:
        _log_once(
            str(exc),
            f"scatter_nd_update_v2: unsupported cache layout ({exc}); raising "
            "so dispatch can fall back to reference.torch",
        )
        raise
    _launch(plan)


# Dispatch-facing op name (``vllm_fl.dispatch.ops`` / ``CachedOp`` key).  The
# implementation itself is named after the AscendC operator it ports; the
# ``deepseek_v4_cache_write`` -> ``paged_cache_write`` interface rename is done
# in the integration branch, not in this PR.
deepseek_v4_cache_write = scatter_nd_update_v2


__all__ = [
    "UnsupportedLayout",
    "deepseek_v4_cache_write",
    "scatter_nd_update_v2",
]
