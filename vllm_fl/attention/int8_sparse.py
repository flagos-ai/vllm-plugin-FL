"""Per-page K min/max metadata for sparse selection over INT8 KV pages."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from vllm_fl.attention import kv_int8

_PAGE = kv_int8.PAGE_TOKENS
_HEADS = kv_int8.NUM_KV_HEADS
_DIM = kv_int8.HEAD_DIM
_FEATURES = _HEADS * _DIM
_SPARSE_PAGE = kv_int8.SPARSE_PAGE_BYTES
_SPARSE_PAGE_HALFS = _SPARSE_PAGE // 2
_SCALE_BASE_HALFS = kv_int8.DATA_BYTES // 2
_METADATA_MIN_HALFS = kv_int8.PAGE_BYTES // 2
_METADATA_MAX_HALFS = _METADATA_MIN_HALFS + _FEATURES


def _validate_sparse_cache(cache: torch.Tensor) -> None:
    if (
        not kv_int8.is_int8_layout(cache)
        or kv_int8.page_stride_bytes(cache) != _SPARSE_PAGE
    ):
        raise ValueError(
            "expected INT8 KV pages with the 9344-byte sparse metadata stride"
        )
    if cache.storage_offset() % 2:
        raise ValueError("INT8 sparse page base must be 2-byte aligned")


def metadata_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-block BF16 K-min and K-max aliases after byte 8320."""
    _validate_sparse_cache(cache)
    storage = cache.untyped_storage()
    base = torch.empty(0, dtype=torch.bfloat16, device=cache.device).set_(
        storage, 0, (storage.nbytes() // 2,), (1,)
    )
    storage_offset = cache.storage_offset() // 2
    shape = (cache.shape[0], _HEADS, _DIM)
    strides = (_SPARSE_PAGE_HALFS, _DIM, 1)
    kmin = torch.as_strided(
        base,
        shape,
        strides,
        storage_offset=storage_offset + _METADATA_MIN_HALFS,
    )
    kmax = torch.as_strided(
        base,
        shape,
        strides,
        storage_offset=storage_offset + _METADATA_MAX_HALFS,
    )
    return kmin, kmax


@triton.jit
def _refresh_page(
    Cache,
    Scales,
    KMin,
    KMax,
    block,
    CACHE_PAGE_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    META_PAGE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SCALE_BASE: tl.constexpr,
):
    token = tl.arange(0, PAGE)
    feature = tl.arange(0, HEADS * DIM)
    head = feature // DIM
    raw = tl.load(
        Cache
        + block * CACHE_PAGE_STRIDE
        + token[:, None] * (HEADS * DIM)
        + feature[None, :]
    )
    scale = tl.load(
        Scales
        + block * SCALE_PAGE_STRIDE
        + SCALE_BASE
        + head[None, :] * PAGE
        + token[:, None]
    ).to(tl.float32)
    # Match the stored INT8 -> BF16 K interpretation, apply its stored scale,
    # then round the reconstructed key to BF16 before reducing each page.
    key = (raw.to(tl.float32) * scale).to(tl.bfloat16)
    key_f32 = key.to(tl.float32)
    lo = tl.min(key_f32, 0).to(tl.bfloat16)
    hi = tl.max(key_f32, 0).to(tl.bfloat16)
    tl.store(KMin + block * META_PAGE_STRIDE + feature, lo)
    tl.store(KMax + block * META_PAGE_STRIDE + feature, hi)


@triton.jit
def _refresh_all_layers(
    Pointers,
    Jobs,
    CACHE_PAGE_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    META_PAGE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SCALE_BASE: tl.constexpr,
):
    job, layer = tl.program_id(0), tl.program_id(1)
    block = tl.load(Jobs + job).to(tl.int64)
    row = Pointers + layer * 4
    cache = tl.load(row).to(tl.pointer_type(tl.int8))
    scales = tl.load(row + 1).to(tl.pointer_type(tl.float16))
    kmin = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
    kmax = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
    _refresh_page(
        cache,
        scales,
        kmin,
        kmax,
        block,
        CACHE_PAGE_STRIDE,
        SCALE_PAGE_STRIDE,
        META_PAGE_STRIDE,
        PAGE,
        HEADS,
        DIM,
        SCALE_BASE,
    )


@triton.jit
def _refresh_from_table(
    Cache,
    Scales,
    KMin,
    KMax,
    Table,
    table_stride: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    META_PAGE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SCALE_BASE: tl.constexpr,
):
    logical_page = tl.program_id(0)
    block = tl.load(Table + logical_page * table_stride).to(tl.int64)
    _refresh_page(
        Cache,
        Scales,
        KMin,
        KMax,
        block,
        CACHE_PAGE_STRIDE,
        SCALE_PAGE_STRIDE,
        META_PAGE_STRIDE,
        PAGE,
        HEADS,
        DIM,
        SCALE_BASE,
    )


class MultiLayerMetadataRefresher:
    """Refresh complete INT8 K pages across all bound layers from block IDs."""

    def __init__(self, caches):
        unique = []
        seen = set()
        for cache in caches:
            if not isinstance(cache, torch.Tensor):
                raise ValueError("every INT8 sparse KV cache must be a tensor")
            _validate_sparse_cache(cache)
            pointer = cache.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                unique.append(cache)
        if not unique:
            raise ValueError("at least one padded INT8 KV cache is required")

        first = unique[0]
        expected_shape = tuple(first.shape)
        device = first.device
        if device.type != "cuda":
            raise ValueError("INT8 sparse metadata refresh requires a CUDA device")
        for cache in unique:
            if (
                tuple(cache.shape) != expected_shape
                or cache.device != device
                or kv_int8.page_stride_bytes(cache) != _SPARSE_PAGE
            ):
                raise ValueError(
                    "all INT8 sparse layers must share shape, layout, and device"
                )

        owners = []
        pointers = []
        for cache in unique:
            scales = kv_int8.half_base(cache)
            kmin, kmax = metadata_views(cache)
            owners.extend((cache, scales, kmin, kmax))
            pointers.append(
                (cache.data_ptr(), scales.data_ptr(), kmin.data_ptr(), kmax.data_ptr())
            )

        self.device = device
        self.num_blocks = int(first.shape[0])
        self.num_layers = len(unique)
        self.pointer_table = torch.tensor(
            pointers, dtype=torch.uint64, device="cpu"
        ).to(device)
        self._owners = tuple(owners)

    def refresh(self, jobs: torch.Tensor, num_jobs: int) -> None:
        """Refresh ``num_jobs`` device int32 physical block IDs for every layer."""
        count = int(num_jobs)
        if count < 0 or count > jobs.numel():
            raise ValueError("num_jobs must fit in the jobs vector")
        if count == 0:
            return
        if (
            jobs.dtype != torch.int32
            or jobs.ndim != 1
            or jobs.device != self.device
            or jobs.stride(0) != 1
        ):
            raise ValueError(
                "jobs must be a contiguous device int32 vector of physical block IDs"
            )
        _refresh_all_layers[(count, self.num_layers)](
            self.pointer_table,
            jobs,
            _SPARSE_PAGE,
            _SPARSE_PAGE_HALFS,
            _SPARSE_PAGE_HALFS,
            _PAGE,
            _HEADS,
            _DIM,
            _SCALE_BASE_HALFS,
            num_warps=4,
            num_stages=2,
        )


def refresh_full_pages(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    num_full_pages: int,
) -> None:
    """Refresh metadata for complete logical pages listed in one device row."""
    _validate_sparse_cache(cache)
    count = int(num_full_pages)
    if count < 0 or count > block_table_row.numel():
        raise ValueError("num_full_pages must fit in the block-table row")
    if count == 0:
        return
    if (
        cache.device.type != "cuda"
        or block_table_row.dtype != torch.int32
        or block_table_row.ndim != 1
        or block_table_row.device != cache.device
    ):
        raise ValueError(
            "full-page refresh requires a CUDA cache and device int32 block-table row"
        )
    kmin, kmax = metadata_views(cache)
    scales = kv_int8.half_base(cache)
    _refresh_from_table[(count,)](
        cache,
        scales,
        kmin,
        kmax,
        block_table_row,
        block_table_row.stride(0),
        _SPARSE_PAGE,
        _SPARSE_PAGE_HALFS,
        _SPARSE_PAGE_HALFS,
        _PAGE,
        _HEADS,
        _DIM,
        _SCALE_BASE_HALFS,
        num_warps=4,
        num_stages=2,
    )
