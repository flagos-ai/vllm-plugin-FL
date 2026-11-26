"""Paged BF16 decode page selection for the 17-token physical cache layout."""

import math

import torch

from vllm.triton_utils import tl, triton

PAGE = 16
PHYSICAL_PAGE = 17
KV_HEADS = 2
HEAD_DIM = 128
Q_HEADS = 16
BUDGET_PAGES = 128
BUDGET_TOKENS = BUDGET_PAGES * PAGE
THRESHOLD = 4096
SINK = 1
RECENT = 4
PAGE_TILE = 64
TOPK_CHUNK = 1024


def padded_view(raw: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Return [K/V, blocks, 16 logical tokens, 2 heads, 128] over 17-slot storage."""
    if raw.dtype == torch.int8:
        raw = raw.view(torch.bfloat16)
    if raw.dtype != torch.bfloat16:
        raise ValueError(f"17-slot cache storage must be BF16 bytes, got {raw.dtype}")
    return raw.view(2, num_blocks, PHYSICAL_PAGE, KV_HEADS, HEAD_DIM)[:, :, :PAGE]


def metadata_views(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias the reserved K-min and K-max vectors in each physical page's tail slot."""
    if (
        kv.dtype != torch.bfloat16
        or kv.ndim != 5
        or kv.shape[0] != 2
        or kv.shape[2:] != (PAGE, KV_HEADS, HEAD_DIM)
    ):
        raise ValueError(
            f"expected [2, blocks, 16, 2, 128] logical view, got {tuple(kv.shape)}"
        )
    if (
        kv.stride(1) != PHYSICAL_PAGE * KV_HEADS * HEAD_DIM
        or kv.stride(2) != KV_HEADS * HEAD_DIM
        or kv.stride(3) != HEAD_DIM
        or kv.stride(4) != 1
    ):
        raise ValueError(
            f"logical view must retain a physical 17-token page stride, got {kv.stride()}"
        )
    blocks = kv.shape[1]
    shape = (blocks, KV_HEADS, HEAD_DIM)
    strides = (kv.stride(1), kv.stride(3), kv.stride(4))
    tail = kv.storage_offset() + PAGE * kv.stride(2)
    return (
        torch.as_strided(kv, shape, strides, tail),
        torch.as_strided(kv, shape, strides, tail + kv.stride(0)),
    )


@triton.jit
def _update_metadata(
    K,
    KMin,
    KMax,
    Slots,
    ks0,
    ks1,
    ks2,
    ks3,
    ms0,
    ms1,
    ms2,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Slots + i)
    if (slot >= 0) & ((slot % PAGE) == (PAGE - 1)):
        block = (slot // PAGE).to(tl.int64)
        token = tl.arange(0, PAGE)
        hd = tl.arange(0, 2 * D)
        head = hd // D
        dim = hd % D
        k = tl.load(
            K
            + block * ks0
            + token[:, None] * ks1
            + head[None, :] * ks2
            + dim[None, :] * ks3
        )
        lo, hi = tl.min(k.to(tl.float32), 0), tl.max(k.to(tl.float32), 0)
        tl.store(
            KMin + block * ms0 + head * ms1 + dim * ms2, lo.to(KMin.dtype.element_ty)
        )
        tl.store(
            KMax + block * ms0 + head * ms1 + dim * ms2, hi.to(KMax.dtype.element_ty)
        )


def update_metadata(
    kcache: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Refresh metadata only for full pages whose final token was just written."""
    if slot_mapping.numel() == 0:
        return
    if kcache.shape[1:] != (PAGE, KV_HEADS, HEAD_DIM) or kcache.dtype != torch.bfloat16:
        raise ValueError(
            f"expected BF16 K cache [blocks,16,2,128], got {tuple(kcache.shape)} {kcache.dtype}"
        )
    if (
        kmin.shape != (kcache.shape[0], KV_HEADS, HEAD_DIM)
        or kmax.shape != kmin.shape
        or kmax.stride() != kmin.stride()
    ):
        raise ValueError("K metadata shape does not match the physical block count")
    if not (kcache.device == kmin.device == kmax.device == slot_mapping.device):
        raise ValueError("K cache, metadata, and slot_mapping must share a device")
    _update_metadata[(slot_mapping.numel(),)](
        kcache,
        kmin,
        kmax,
        slot_mapping,
        *kcache.stride(),
        *kmin.stride(),
        slot_mapping.numel(),
        PAGE=PAGE,
        D=HEAD_DIM,
        num_warps=4,
        num_stages=2,
        pipeline="basic",
    )


@triton.jit
def _refresh_metadata_all_layers(
    Pointers,
    Jobs,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    job, layer = tl.program_id(0), tl.program_id(1)
    block = tl.load(Jobs + job).to(tl.int64)
    k_ptr = tl.load(Pointers + layer * 3).to(tl.pointer_type(tl.bfloat16))
    min_ptr = tl.load(Pointers + layer * 3 + 1).to(tl.pointer_type(tl.bfloat16))
    max_ptr = tl.load(Pointers + layer * 3 + 2).to(tl.pointer_type(tl.bfloat16))

    token = tl.arange(0, PAGE)
    feature = tl.arange(0, 2 * D)
    values = tl.load(
        k_ptr + block * CACHE_BLOCK_STRIDE + token[:, None] * (2 * D) + feature[None, :]
    ).to(tl.float32)
    lo = tl.min(values, 0).to(tl.bfloat16)
    hi = tl.max(values, 0).to(tl.bfloat16)
    tl.store(min_ptr + block * CACHE_BLOCK_STRIDE + feature, lo)
    tl.store(max_ptr + block * CACHE_BLOCK_STRIDE + feature, hi)


class MultiLayerMetadataRefresher:
    """Refresh full physical pages across all bound BF16 attention layers."""

    def __init__(self, kv_caches: list[torch.Tensor]):
        if not kv_caches:
            raise ValueError("at least one padded KV cache is required")

        unique_caches = []
        seen_ptrs: set[int] = set()
        for kv in kv_caches:
            if not isinstance(kv, torch.Tensor):
                raise ValueError("every native sparse KV cache must be a tensor")
            if kv.data_ptr() in seen_ptrs:
                continue
            seen_ptrs.add(kv.data_ptr())
            unique_caches.append(kv)

        first = unique_caches[0]
        if first.dtype != torch.bfloat16 or first.ndim != 5 or first.shape[0] != 2:
            raise ValueError(
                f"expected BF16 [2,blocks,16,2,128] KV cache, got {tuple(first.shape)} {first.dtype}"
            )
        blocks = int(first.shape[1])
        expected_shape = (2, blocks, PAGE, KV_HEADS, HEAD_DIM)
        expected_stride = (
            blocks * PHYSICAL_PAGE * KV_HEADS * HEAD_DIM,
            PHYSICAL_PAGE * KV_HEADS * HEAD_DIM,
            KV_HEADS * HEAD_DIM,
            HEAD_DIM,
            1,
        )
        if tuple(first.shape) != expected_shape or first.stride() != expected_stride:
            raise ValueError(
                "native sparse refresh requires the 17-slot KV shape/stride "
                f"{expected_shape}/{expected_stride}, got {tuple(first.shape)}/{first.stride()}"
            )

        device = first.device
        if device.type != "cuda":
            raise ValueError("native sparse metadata refresh requires a CUDA device")
        pointers = []
        for kv in unique_caches:
            if (
                kv.dtype != torch.bfloat16
                or tuple(kv.shape) != expected_shape
                or kv.stride() != expected_stride
                or kv.device != device
            ):
                raise ValueError(
                    "all native sparse KV layers must share shape, stride, dtype, and device"
                )
            kcache = kv[0]
            kmin, kmax = metadata_views(kv)
            pointers.append((kcache.data_ptr(), kmin.data_ptr(), kmax.data_ptr()))

        self.device = device
        self.num_blocks = blocks
        self.num_layers = len(pointers)
        # Keep only the pointer table; the model runner owns KV tensor lifetimes.
        self.pointer_table = torch.tensor(pointers, dtype=torch.uint64).to(
            device=device
        )

    def refresh(self, jobs: torch.Tensor, num_jobs: int) -> None:
        """Refresh each listed full physical page in each bound layer."""
        count = int(num_jobs)
        if count == 0:
            return
        if (
            jobs.dtype != torch.int32
            or jobs.ndim != 1
            or jobs.device != self.device
            or jobs.stride(0) != 1
            or count < 0
            or count > jobs.numel()
        ):
            raise ValueError(
                "jobs must be a contiguous device int32 vector containing num_jobs entries"
            )
        _refresh_metadata_all_layers[(count, self.num_layers)](
            self.pointer_table,
            jobs,
            PHYSICAL_PAGE * KV_HEADS * HEAD_DIM,
            PAGE=PAGE,
            D=HEAD_DIM,
            num_warps=4,
            num_stages=2,
            pipeline="basic",
        )


@triton.jit
def _page_scores(
    Q,
    KMin,
    KMax,
    Table,
    Seq,
    Scores,
    q0,
    q1,
    q2,
    lo0,
    lo1,
    lo2,
    hi0,
    hi1,
    hi2,
    bt0,
    bt1,
    s0,
    s1,
    s2,
    DENSE_LIMIT: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    PT: tl.constexpr,
    GRID_P: tl.constexpr,
    SINK: tl.constexpr,
    RECENT: tl.constexpr,
):
    group, row, kvh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = tl.load(Seq + row)
    if seq > DENSE_LIMIT:
        npages = tl.cdiv(seq, PAGE)
        full_pages = seq // PAGE
        for chunk in tl.range(group, tl.cdiv(npages, PT), GRID_P, num_stages=2):
            p = chunk * PT + tl.arange(0, PT)
            eligible = (
                (p >= SINK) & (p < npages - RECENT) & (p < full_pages) & (p < MAX_PAGES)
            )
            block = tl.load(Table + row.to(tl.int64) * bt0 + p * bt1, eligible, 0).to(
                tl.int64
            )
            r = tl.arange(0, 16)
            dk = tl.arange(0, BK)
            accum = tl.zeros((16, PT), tl.float32)
            for dblock in tl.range(0, tl.cdiv(D, BK), num_stages=2):
                d = dblock * BK + dk
                q = tl.load(
                    Q
                    + row.to(tl.int64) * q0
                    + (kvh * 8 + r[:, None]) * q1
                    + d[None, :] * q2,
                    (r[:, None] < 8) & (d[None, :] < D),
                    0,
                ).to(tl.bfloat16)
                qpos = tl.where(q > 0, q, tl.full((), 0, tl.bfloat16))
                qneg = tl.where(q < 0, q, tl.full((), 0, tl.bfloat16))
                kp = tl.load(
                    KMax + block[:, None] * hi0 + kvh * hi1 + d[None, :] * hi2,
                    eligible[:, None] & (d[None, :] < D),
                    0,
                )
                kn = tl.load(
                    KMin + block[:, None] * lo0 + kvh * lo1 + d[None, :] * lo2,
                    eligible[:, None] & (d[None, :] < D),
                    0,
                )
                accum = tl.dot(qpos, tl.trans(kp), accum)
                accum = tl.dot(qneg, tl.trans(kn), accum)
            accum = tl.where(r[:, None] < 8, accum, -float("inf"))
            page_score = tl.max(accum, 0)
            tl.store(
                Scores + row.to(tl.int64) * s0 + kvh * s1 + p * s2, page_score, eligible
            )


@triton.jit
def _float_ordered_key(score, page):
    score = tl.where(score == 0.0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    low = 0xFFFFFFFF - page.to(tl.uint32)
    return (ordered.to(tl.uint64) << 32) | low.to(tl.uint64)


@triton.jit
def _topk_partial(
    Scores,
    Seq,
    Partial,
    s0,
    s1,
    s2,
    p0,
    p1,
    p2,
    MAX_PAGES: tl.constexpr,
    DENSE_LIMIT: tl.constexpr,
    PAGE: tl.constexpr,
    SINK: tl.constexpr,
    RECENT: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    base = chunk * BLOCK
    seq = tl.load(Seq + row)
    npages = tl.cdiv(seq, PAGE)
    x = tl.arange(0, BLOCK)
    page = base + x
    if (seq > DENSE_LIMIT) & (base < npages):
        valid = (page < npages) & (page < MAX_PAGES)
        forced = valid & ((page < SINK) | (page >= npages - RECENT))
        candidate = valid & ~forced
        s0v = tl.load(Scores + row.to(tl.int64) * s0 + page * s2, candidate, 0.0)
        s1v = tl.load(Scores + row.to(tl.int64) * s0 + s1 + page * s2, candidate, 0.0)
        score = tl.maximum(s0v, s1v)
        key = _float_ordered_key(score, page)
        key = tl.where(
            forced, _float_ordered_key(float("inf"), page), tl.where(valid, key, 0)
        )
        sorted_key = tl.sort(key, dim=0, descending=True)
        top = tl.gather(sorted_key, tl.arange(0, K), axis=0)
        tl.store(
            Partial + row.to(tl.int64) * p0 + chunk * p1 + tl.arange(0, K) * p2, top
        )


@triton.jit
def _merge_candidates(
    Partial, row, p0, p1, p2, live_chunks, WIDTH: tl.constexpr, TOPK: tl.constexpr
):
    i = tl.arange(0, WIDTH)
    keys = tl.load(
        Partial + row.to(tl.int64) * p0 + (i // TOPK) * p1 + (i % TOPK) * p2,
        i < live_chunks * TOPK,
        0,
    )
    sorted_keys = tl.sort(keys, dim=0, descending=True)
    return tl.gather(sorted_keys, tl.arange(0, TOPK), axis=0)


@triton.jit
def _compact_topk(
    Partial,
    Table,
    Seq,
    Selected,
    Lengths,
    p0,
    p1,
    p2,
    bt0,
    bt1,
    out0,
    out1,
    MAX_PAGES: tl.constexpr,
    NCHUNKS: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_PAD: tl.constexpr,
    DENSE_LIMIT: tl.constexpr,
    PAGE: tl.constexpr,
    BUDGET_PAGES: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    seq = tl.load(Seq + row)
    npages = tl.cdiv(seq, PAGE)
    if seq <= DENSE_LIMIT:
        dense_j = tl.arange(0, OUT_PAD)
        block = tl.load(
            Table + row.to(tl.int64) * bt0 + dense_j * bt1,
            (dense_j < npages) & (dense_j < MAX_PAGES),
            0,
        )
        tl.store(
            Selected + row.to(tl.int64) * out0 + dense_j * out1,
            tl.where(dense_j < npages, block, 0),
            dense_j < OUT_WIDTH,
        )
        tl.store(Lengths + row, seq)
    else:
        live_chunks = tl.minimum(tl.cdiv(npages, 1024), NCHUNKS)
        offs = tl.arange(0, TOPK)
        if live_chunks == 1:
            best = tl.load(Partial + row.to(tl.int64) * p0 + offs * p2)
        elif live_chunks <= 2:
            best = _merge_candidates(
                Partial, row, p0, p1, p2, live_chunks, WIDTH=2 * TOPK, TOPK=TOPK
            )
        elif live_chunks <= 4:
            best = _merge_candidates(
                Partial, row, p0, p1, p2, live_chunks, WIDTH=4 * TOPK, TOPK=TOPK
            )
        else:
            best = _merge_candidates(
                Partial, row, p0, p1, p2, live_chunks, WIDTH=8 * TOPK, TOPK=TOPK
            )
        logical = (0xFFFFFFFF - best.to(tl.uint32)).to(tl.int32)
        logical = tl.sort(logical, dim=0, descending=False)
        logical_valid = (logical >= 0) & (logical < npages) & (logical < MAX_PAGES)
        safe_logical = tl.where(logical_valid, logical, 0)
        physical = tl.load(
            Table + row.to(tl.int64) * bt0 + safe_logical * bt1, logical_valid, other=-1
        )
        selected_j = tl.arange(0, TOPK)
        tl.store(Selected + row.to(tl.int64) * out0 + selected_j * out1, physical)
        tail = seq - (npages - 1) * PAGE
        tl.store(Lengths + row, (BUDGET_PAGES - 1) * PAGE + tail)


class Workspace:
    """One reusable selection workspace shared by every model layer."""

    def __init__(
        self,
        device,
        max_seqs: int,
        max_pages: int,
        dense_threshold: int = THRESHOLD,
        budget_tokens: int = BUDGET_TOKENS,
        *,
        pipeline: str | None = "basic",
    ):
        if max_seqs <= 0 or max_pages <= 0 or max_pages > 8192 or dense_threshold < 0:
            raise ValueError(
                "max_seqs must be positive, max_pages in [1,8192], and dense_threshold nonnegative"
            )
        if budget_tokens != BUDGET_TOKENS:
            raise ValueError(f"this selector uses a fixed {BUDGET_TOKENS}-token budget")
        self.device = torch.device(device)
        self.max_seqs, self.max_pages = max_seqs, max_pages
        self.dense_threshold, self.budget_tokens = dense_threshold, budget_tokens
        self.dense_limit = max(dense_threshold, budget_tokens)
        self.num_chunks = math.ceil(max_pages / TOPK_CHUNK)
        self.selected_width = max(math.ceil(dense_threshold / PAGE), BUDGET_PAGES)
        self.out_pad = 1 << (self.selected_width - 1).bit_length()
        self.scores = torch.empty(
            (max_seqs, KV_HEADS, max_pages), dtype=torch.float32, device=self.device
        )
        self.device = self.scores.device
        self.launch_options = {"pipeline": pipeline} if pipeline is not None else {}
        self.partial_keys = torch.empty(
            (max_seqs, self.num_chunks, BUDGET_PAGES),
            dtype=torch.uint64,
            device=self.device,
        )
        self.selected = torch.empty(
            (max_seqs, self.selected_width), dtype=torch.int32, device=self.device
        )
        self.compact_lengths = torch.empty(
            (max_seqs,), dtype=torch.int32, device=self.device
        )

    @property
    def nbytes(self) -> int:
        return sum(
            x.numel() * x.element_size()
            for x in (
                self.scores,
                self.partial_keys,
                self.selected,
                self.compact_lengths,
            )
        )


def compact(
    query: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace: Workspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select pages for single-token decode, returning fixed-address buffers safe for CUDA Graphs."""
    if (
        query.ndim != 3
        or tuple(query.shape[1:]) != (Q_HEADS, HEAD_DIM)
        or query.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"query must be BF16 [B,16,128], got {tuple(query.shape)} {query.dtype}"
        )
    batch = query.shape[0]
    if batch > workspace.max_seqs or table.shape != (batch, workspace.max_pages):
        raise ValueError("batch/table shape exceeds or differs from Workspace capacity")
    if (
        seq_lens.shape != (batch,)
        or seq_lens.dtype != torch.int32
        or table.dtype != torch.int32
    ):
        raise ValueError(
            "seq_lens and table must be int32 with matching batch dimensions"
        )
    if (
        kmin.ndim != 3
        or tuple(kmin.shape[1:]) != (KV_HEADS, HEAD_DIM)
        or kmax.shape != kmin.shape
    ):
        raise ValueError("kmin/kmax must have shape [physical_blocks,2,128]")
    if not all(
        t.device == workspace.device for t in (query, kmin, kmax, table, seq_lens)
    ):
        raise ValueError(
            "query, metadata, table, seq_lens, and Workspace must share a device"
        )
    if batch == 0:
        return workspace.selected[:0], workspace.compact_lengths[:0]

    _page_scores[(4, batch, KV_HEADS)](
        query,
        kmin,
        kmax,
        table,
        seq_lens,
        workspace.scores,
        *query.stride(),
        *kmin.stride(),
        *kmax.stride(),
        *table.stride(),
        *workspace.scores.stride(),
        workspace.dense_limit,
        workspace.max_pages,
        PAGE=PAGE,
        D=HEAD_DIM,
        BK=64,
        PT=PAGE_TILE,
        GRID_P=4,
        SINK=SINK,
        RECENT=RECENT,
        num_warps=4,
        num_stages=2,
        **workspace.launch_options,
    )
    _topk_partial[(batch, workspace.num_chunks)](
        workspace.scores,
        seq_lens,
        workspace.partial_keys,
        *workspace.scores.stride(),
        *workspace.partial_keys.stride(),
        workspace.max_pages,
        workspace.dense_limit,
        PAGE=PAGE,
        SINK=SINK,
        RECENT=RECENT,
        BLOCK=TOPK_CHUNK,
        K=BUDGET_PAGES,
        num_warps=4,
        num_stages=2,
        **workspace.launch_options,
    )
    _compact_topk[(batch,)](
        workspace.partial_keys,
        table,
        seq_lens,
        workspace.selected,
        workspace.compact_lengths,
        *workspace.partial_keys.stride(),
        *table.stride(),
        *workspace.selected.stride(),
        workspace.max_pages,
        workspace.num_chunks,
        workspace.selected_width,
        workspace.out_pad,
        workspace.dense_limit,
        PAGE,
        BUDGET_PAGES,
        BUDGET_PAGES,
        num_warps=4,
        num_stages=2,
        **workspace.launch_options,
    )
    return workspace.selected[:batch], workspace.compact_lengths[:batch]
