# SPDX-License-Identifier: Apache-2.0
"""Triton paged-cache row writer with stride and padding support."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

_DEFAULT_NUM_PROGRAMS = 96
_MAX_BLOCK_WIDTH = 1024
_INT32_MAX = 2**31 - 1
_SUPPORTED_CACHE_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int8)


@triton.jit
def _cache_write_kernel(
    VALUES,  # *values_dtype [tokens, width] contiguous
    INDICES,  # *int32/int64 [tokens, 2] contiguous
    CACHE,  # *cache_dtype; base pointer == cache.data_ptr()
    TOKENS,  # runtime int
    WIDTH,  # runtime int, row width == cache.shape[-1]
    STRIDE0,  # cache.stride(0)
    STRIDE1,  # cache.stride(1)
    STRIDE2,  # cache.stride(2)
    PAGES,  # cache.shape[0]
    BLOCK,  # cache.shape[1]
    LIMIT,  # storage elements addressable from cache.data_ptr()
    NC: tl.constexpr,
    BLK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    stride2 = tl.cast(STRIDE2, tl.int64)
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
            & (address + (WIDTH - 1) * stride2 < LIMIT)
        )
        address = tl.minimum(tl.maximum(address, 0), LIMIT - 1)
        for start in range(0, tl.cdiv(WIDTH, BLK_W)):
            cols = start * BLK_W + tl.arange(0, BLK_W)
            mask = valid & (cols < WIDTH)
            source = tl.load(VALUES + row * WIDTH + cols, mask=mask, other=0)
            tl.store(CACHE + address + cols * stride2, source, mask=mask)


class _Unsupported(NotImplementedError):
    """The layout cannot be written safely by this backend."""


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
    pages: int
    block: int
    limit: int


#: Cache for the (pages, block, stride0, stride1, width) alias analysis.
_ADDRESS_MAP_SAFE_CACHE: dict[tuple[int, int, int, int, int], bool] = {}


def _storage_limit(cache: torch.Tensor) -> int:
    """Elements addressable from ``cache.data_ptr()`` to the storage end."""
    try:
        storage_bytes = int(cache.untyped_storage().nbytes())
        element_size = int(cache.element_size())
        storage_offset = int(cache.storage_offset())
    except Exception as exc:  # noqa: BLE001 - layout without a usable storage view
        raise _Unsupported("storage_unknown") from exc
    if element_size <= 0:
        raise _Unsupported("storage_unknown")
    limit = storage_bytes // element_size - storage_offset
    if limit <= 0:
        raise _Unsupported("storage_unknown")
    return limit


def _address_map_is_safe(
    pages: int, block: int, stride0: int, stride1: int, width: int
) -> bool:
    """True iff two different valid slots cannot touch overlapping rows.

    Caller-guaranteed unique slots only imply disjoint writes when the slot ->
    address map is injective at row granularity.  When the two leading strides
    are multiples of the row width, rows start on width boundaries, so the
    remaining alias source is an overlapping page view: the last row of one page
    must end before the next page starts.  A rare misaligned or overlapping
    layout is proven (or rejected) by looking for a non-zero
    ``page_delta * stride0 + offset_delta * stride1`` shorter than one row.
    """
    page_span = (block - 1) * stride1 + width
    pages_disjoint = pages <= 1 or page_span <= stride0
    if pages_disjoint and (block <= 1 or stride1 >= width):
        return True
    key = (pages, block, stride0, stride1, width)
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
            if abs(page_delta * stride0 + offset_delta * stride1) < width:
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
        raise _Unsupported("slots_dtype")
    return torch.stack((slots // block, slots % block), dim=-1)


def _plan(values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor) -> _WritePlan:
    """Validate the storage layout before launching any cache writes."""
    pages, block, width = (int(size) for size in cache.shape)
    stride0, stride1, stride2 = (int(stride) for stride in cache.stride())
    if stride0 <= 0 or stride1 <= 0 or stride2 <= 0:
        raise _Unsupported("negative_or_zero_stride")
    limit = _storage_limit(cache)
    if pages * block > _INT32_MAX:
        raise _Unsupported("num_slots_too_large")
    if not _address_map_is_safe(
        pages, block, stride0, stride1, (width - 1) * stride2 + 1
    ):
        raise _Unsupported("address_map_alias")
    if values.dtype != cache.dtype:
        # The reference writer stores values in the cache dtype; keep bit parity.
        values = values.to(cache.dtype)
    if not values.is_contiguous():
        values = values.contiguous()
    if values.numel() > _INT32_MAX:
        raise _Unsupported("values_too_large")
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
        pages=pages,
        block=block,
        limit=limit,
    )


def _num_programs() -> int:
    """Use two programs per vector core."""
    try:
        from .triton_utils import get_vectorcore_num

        return max(1, 2 * int(get_vectorcore_num()))
    except Exception:  # noqa: BLE001 - device properties not initialized
        return _DEFAULT_NUM_PROGRAMS


def _launch(plan: _WritePlan) -> None:
    num_programs = _num_programs()
    block_width = min(triton.next_power_of_2(plan.width), _MAX_BLOCK_WIDTH)
    _cache_write_kernel[(num_programs,)](
        plan.values,
        plan.indices,
        plan.cache,
        plan.tokens,
        plan.width,
        plan.stride0,
        plan.stride1,
        plan.stride2,
        plan.pages,
        plan.block,
        plan.limit,
        NC=num_programs,
        BLK_W=block_width,
    )


def _check_contract(
    values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor
) -> None:
    """Validate the paged-cache tensor contract."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    if values.device != cache.device or slots.device != cache.device:
        raise ValueError("Values, slots and cache must be on the same device")
    if cache.device.type != "npu":
        raise ValueError("Ascend paged cache writes require NPU tensors")
    if cache.dtype not in _SUPPORTED_CACHE_DTYPES:
        raise ValueError("Unsupported cache storage dtype")


def scatter_nd_update_v2(
    values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor
) -> None:
    """Mutate [pages, block, dim] cache at unique slots, skipping padding.

    Values are [tokens, dim], slots are int32/int64 [tokens]. Negative and
    out-of-range slots are skipped. Positive, disjoint cache strides and
    storage-offset views are supported; cache storage is never copied.
    """
    _check_contract(values, slots, cache)
    if values.shape[0] == 0:
        return
    if any(size == 0 for size in cache.shape):
        raise ValueError("Cannot write to an empty cache")
    _launch(_plan(values, slots, cache))
