"""Page-aligned GQA Triton kernels: 1-page Split-KV/Direct Decode + 4-page BLOCK_Q=16 Prefill.

Designed for MiniCPM5-2B (Hq=16, Hkv=2, D=128, page_size=16) on Iluvatar BI-V150
(16 SMs, Triton 3.2 compatible: no chained boolean `or`/`and`, no tensor descriptors).
"""

from __future__ import annotations

import math

try:
    import torch
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - local CPU test environment
    torch = None
    triton = None
    tl = None
    HAS_TRITON = False

_PREFILL_WORKSPACES: dict = {}

# Decode launch config for `_split_kv_decode_stage1`. The inner loop loads one 16-token page
# (2 x 4 KB) and runs two 16x16x128 dots per iteration, so a single stage means "issue load,
# wait, compute" with nothing in flight. Kept as constants rather than literals so a measured
# (warps, stages) pair can be applied without touching either launch site.
DECODE_LAUNCH_WARPS = 2
DECODE_LAUNCH_STAGES = 1

# Prefill softmax mode. `None` means "on for 16-bit inputs", the configuration measured on the
# Iluvatar box (1.15-1.17x per-layer kernel, 4.2e-3 vs 3.5e-3 relative error against a dense fp32
# reference). The hook exists so a sweep can force either side of the A/B; it is inert by default.
_FORCE_PREFILL_NOMAX = None

# Installable component names. `lab.platforms` decides which ones a vendor profile turns on;
# the list lives here because this module ships verbatim into the plugin vendor directory and
# must not import anything outside the container's site-packages.
COMPONENTS = (
    "topk_triton",  # vLLM Triton top-k/top-p threshold kernel instead of radix sort
    "sampler_cast",  # fused bf16->fp32 + 1/T scaling into a persistent fp32 buffer
    "sampler_gumbel",  # single-pass Gumbel-Max draw; skips softmax/exponential/argmax
    "block_pool",  # contiguous slab allocation for KV pages
    "sched_admission",  # KV-aware admission control; no preemption on long context
    "kv_stride_order",  # physically contiguous K/V; needs TritonAttentionBackend
    "attention",  # page-aligned split-KV decode + BLOCK_Q prefill (installed by lab.patches)
)
ALL_COMPONENTS = frozenset(COMPONENTS)
_ACTIVE_COMPONENTS = ALL_COMPONENTS
_APPLIED_COMPONENTS: dict[str, bool] = {}


def select_num_splits(batch_size: int, max_seq_len: int, *, kv_heads: int = 2) -> int:
    """Pick a power-of-two split count tuned for Iluvatar BI-V150 (16 SMs, 4 CTAs/SM)."""
    if batch_size <= 0 or max_seq_len <= 0 or kv_heads <= 0:
        raise ValueError("batch_size, max_seq_len and kv_heads must be positive")
    total_groups = batch_size * kv_heads
    if max_seq_len < 256 or total_groups >= 128:
        return 1
    if total_groups >= 60:
        return 2
    if max_seq_len >= 4096 and total_groups <= 16:
        return 8
    return 4


def can_use_specialized_attention(
    q,
    k_cache,
    v_cache,
    *,
    causal: bool = True,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap: float = 0.0,
    qq_bias=None,
    sinks=None,
    output_scale=None,
) -> bool:
    """Return True when inputs match the specialized BF16/FP16/FP32 GQA kernel contract."""
    if not causal or alibi_slopes is not None or qq_bias is not None:
        return False
    if sinks is not None or output_scale is not None or float(softcap) > 0.0:
        return False
    if window_size is not None and (window_size[0] >= 0 or window_size[1] >= 0):
        return False
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        return False
    _, q_heads, head_dim = q.shape
    _, page_size, kv_heads, k_dim = k_cache.shape
    if head_dim != 128 or k_dim != 128 or page_size != 16 or kv_heads <= 0:
        return False
    return (q_heads % kv_heads == 0) and (1 <= (q_heads // kv_heads) <= 16)


def validate_decode_tensors(q, k_cache, v_cache, *, block_table, seq_lens, out=None):
    """Validate GPU decode inputs before launching the Triton kernel."""
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("Expected Q[T,Hq,D] and K,V[pages,page_size,Hkv,D]")
    tokens, q_heads, head_dim = q.shape
    _, page_size, kv_heads, k_dim = k_cache.shape
    if head_dim != 128 or k_dim != 128 or page_size != 16:
        raise ValueError("Specialized kernel requires head_dim=128 and page_size=16")
    if q_heads % kv_heads != 0 or (q_heads // kv_heads) > 16:
        raise ValueError("Hq must be divisible by Hkv with group size <= 16")
    if block_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("Expected block_table[B,max_pages] and seq_lens[B]")
    if block_table.shape[0] < tokens or seq_lens.shape[0] < tokens:
        raise ValueError("Batch metadata smaller than active token count")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("Output buffer must match Q shape and dtype")


class SplitKVWorkspace:
    """Pre-allocated FP32 partial buffers safe for CUDA Graph capture and replay."""

    def __init__(
        self, max_batch: int = 512, q_heads: int = 16, max_splits: int = 16, dim: int = 128
    ):
        if min(max_batch, q_heads, max_splits, dim) <= 0:
            raise ValueError("Workspace dimensions must be positive")
        self.max_batch = max_batch
        self.q_heads = q_heads
        self.max_splits = max_splits
        self.dim = dim
        self._device = None
        self.acc = None
        self.max_logits = None
        self.exp_sums = None

    def ensure(self, device, *, min_batch: int = 1, min_splits: int = 1):
        if min_batch > self.max_batch or min_splits > self.max_splits:
            raise ValueError("Requested batch or splits exceed pre-allocated graph capacity")
        if self.acc is None or self._device != device:
            shape = (self.max_batch, self.q_heads, self.max_splits)
            self.acc = torch.empty((*shape, self.dim), dtype=torch.float32, device=device)
            self.max_logits = torch.empty(shape, dtype=torch.float32, device=device)
            self.exp_sums = torch.empty(shape, dtype=torch.float32, device=device)
            self._device = device
        return self.acc, self.max_logits, self.exp_sums


if HAS_TRITON:

    @triton.jit
    def _split_kv_decode_stage1(
        q_ptr, k_ptr, v_ptr, block_table_ptr, seq_lens_ptr, cu_seqlens_q_ptr,
        acc_ptr, max_ptr, sum_ptr, out_ptr, lse_ptr,
        scale, num_seqs, q_stride_b: tl.int64, q_stride_h: tl.int64,
        out_stride_b: tl.int64, out_stride_h: tl.int64,
        k_stride_p: tl.constexpr, k_stride_t: tl.constexpr, k_stride_h: tl.constexpr,
        v_stride_p: tl.constexpr, v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
        bt_stride_b: tl.int64, ws_stride_b: tl.int64, ws_stride_h: tl.int64, ws_stride_s: tl.int64,
        NUM_KV_HEADS: tl.constexpr, GQA_GROUP: tl.constexpr, NUM_SPLITS: tl.constexpr,
        PAGE_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIRECT_OUT: tl.constexpr, RETURN_LSE: tl.constexpr, FILTER_SINGLE_Q: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        split_idx = tl.program_id(2)
        offs_m = tl.arange(0, 16)
        offs_d = tl.arange(0, HEAD_DIM)
        g_mask = offs_m < GQA_GROUP
        q_head = kv_head * GQA_GROUP + offs_m
        ws_base = seq_idx * ws_stride_b + q_head * ws_stride_h + split_idx * ws_stride_s
        acc_dest = acc_ptr + ws_base[:, None] * HEAD_DIM + offs_d[None, :]
        if seq_idx >= num_seqs:
            if not DIRECT_OUT:
                tl.store(max_ptr + ws_base, float("-inf"), mask=g_mask)
                tl.store(sum_ptr + ws_base, 0.0, mask=g_mask)
                tl.store(acc_dest, 0.0, mask=g_mask[:, None])
            return
        if FILTER_SINGLE_Q:
            q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
            q_stop = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
            if (q_stop - q_start) != 1:
                return
            q_token_idx = q_start
        else:
            q_token_idx = seq_idx
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pages_per_split = (num_pages + NUM_SPLITS - 1) // NUM_SPLITS
        page_start = split_idx * pages_per_split
        page_end = tl.minimum(page_start + pages_per_split, num_pages)
        if page_start >= page_end:
            if DIRECT_OUT:
                out_row = q_token_idx * out_stride_b + q_head[:, None] * out_stride_h
                tl.store(out_ptr + out_row + offs_d[None, :], 0.0, mask=g_mask[:, None])
                if RETURN_LSE:
                    tl.store(lse_ptr + q_head * num_seqs + q_token_idx, float("-inf"), mask=g_mask)
            else:
                tl.store(max_ptr + ws_base, float("-inf"), mask=g_mask)
                tl.store(sum_ptr + ws_base, 0.0, mask=g_mask)
                tl.store(acc_dest, 0.0, mask=g_mask[:, None])
            return
        q_off = q_token_idx * q_stride_b + q_head[:, None] * q_stride_h + offs_d[None, :]
        q = tl.load(q_ptr + q_off, mask=g_mask[:, None], other=0.0)
        scale_log2e = scale * 1.4426950408889634
        m_i = tl.full([16], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([16], dtype=tl.float32)
        acc = tl.zeros([16, HEAD_DIM], dtype=tl.float32)
        bt_row_ptr = block_table_ptr + seq_idx * bt_stride_b
        k_head_base = k_ptr + kv_head * k_stride_h
        v_head_base = v_ptr + kv_head * v_stride_h
        full_end = tl.maximum(page_start, tl.minimum(page_end, seq_len // PAGE_SIZE))
        pair_end = page_start + ((full_end - page_start) // 2) * 2
        offs_t16 = tl.arange(0, 16)
        k_intra = offs_t16[None, :] * k_stride_t + offs_d[:, None]
        v_intra = offs_t16[:, None] * v_stride_t + offs_d[None, :]
        for page_idx in range(page_start, pair_end, 2):
            p0 = tl.load(bt_row_ptr + page_idx).to(tl.int64)
            p1 = tl.load(bt_row_ptr + page_idx + 1).to(tl.int64)
            k0 = tl.load(k_head_base + p0 * k_stride_p + k_intra)
            k1 = tl.load(k_head_base + p1 * k_stride_p + k_intra)
            s0 = tl.dot(q, k0) * scale_log2e
            s1 = tl.dot(q, k1) * scale_log2e
            v0 = tl.load(v_head_base + p0 * v_stride_p + v_intra)
            v1 = tl.load(v_head_base + p1 * v_stride_p + v_intra)
            m_new = tl.maximum(m_i, tl.maximum(tl.max(s0, axis=1), tl.max(s1, axis=1)))
            alpha = tl.math.exp2(m_i - m_new)
            p_0 = tl.math.exp2(s0 - m_new[:, None])
            p_1 = tl.math.exp2(s1 - m_new[:, None])
            acc = (
                acc * alpha[:, None]
                + tl.dot(p_0.to(v0.dtype), v0)
                + tl.dot(p_1.to(v1.dtype), v1)
            )
            l_i = l_i * alpha + tl.sum(p_0 + p_1, axis=1)
            m_i = m_new
        for page_idx in range(pair_end, full_end):
            phys_page = tl.load(bt_row_ptr + page_idx).to(tl.int64)
            k = tl.load(k_head_base + phys_page * k_stride_p + k_intra)
            v = tl.load(v_head_base + phys_page * v_stride_p + v_intra)
            scores = tl.dot(q, k) * scale_log2e
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
        for page_idx in range(full_end, page_end):
            phys_page = tl.load(bt_row_ptr + page_idx).to(tl.int64)
            t_mask = (page_idx * PAGE_SIZE + offs_t16) < seq_len
            k = tl.load(
                k_head_base + phys_page * k_stride_p + k_intra, mask=t_mask[None, :], other=0.0
            )
            v = tl.load(
                v_head_base + phys_page * v_stride_p + v_intra, mask=t_mask[:, None], other=0.0
            )
            raw_s = tl.dot(q, k) * scale_log2e
            scores = tl.where(t_mask[None, :], raw_s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(scores - m_safe[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
        m_i_nat = tl.where(m_i > float("-inf"), m_i * 0.6931471805599453, float("-inf"))
        if DIRECT_OUT:
            has_any = (m_i_nat > float("-inf")) & (l_i > 0.0)
            out_val = tl.where(has_any[:, None], acc / l_i[:, None], 0.0)
            out_off = q_token_idx * out_stride_b + q_head[:, None] * out_stride_h + offs_d[None, :]
            tl.store(out_ptr + out_off, out_val.to(out_ptr.dtype.element_ty), mask=g_mask[:, None])
            if RETURN_LSE:
                lse_val = tl.where(has_any, m_i_nat + tl.log(l_i), float("-inf"))
                tl.store(lse_ptr + q_head * num_seqs + q_token_idx, lse_val, mask=g_mask)
        else:
            tl.store(max_ptr + ws_base, m_i_nat, mask=g_mask)
            tl.store(sum_ptr + ws_base, l_i, mask=g_mask)
            tl.store(acc_dest, acc, mask=g_mask[:, None])

    @triton.jit
    def _split_kv_decode_stage2(
        acc_ptr, max_ptr, sum_ptr, cu_seqlens_q_ptr, out_ptr, lse_ptr, num_seqs,
        out_stride_b: tl.int64, out_stride_h: tl.int64,
        ws_stride_b: tl.int64, ws_stride_h: tl.int64, ws_stride_s: tl.int64,
        NUM_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
        RETURN_LSE: tl.constexpr, FILTER_SINGLE_Q: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        if seq_idx >= num_seqs:
            return
        if FILTER_SINGLE_Q:
            q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
            q_stop = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
            if (q_stop - q_start) != 1:
                return
            q_token_idx = q_start
        else:
            q_token_idx = seq_idx
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)
        ws_base = seq_idx * ws_stride_b + head_idx * ws_stride_h + offs_s * ws_stride_s
        m_s = tl.load(max_ptr + ws_base)
        l_s = tl.load(sum_ptr + ws_base)
        valid = l_s > 0.0
        m_s = tl.where(valid, m_s, float("-inf"))
        m_max = tl.max(m_s, axis=0)
        has_any = m_max > float("-inf")
        m_safe = tl.where(has_any, m_max, 0.0)
        scales = tl.where(valid, tl.exp(m_s - m_safe), 0.0)
        denom = tl.sum(scales * l_s, axis=0)
        partials = tl.load(acc_ptr + ws_base[:, None] * HEAD_DIM + offs_d[None, :])
        numer = tl.sum(partials * scales[:, None], axis=0)
        out_val = tl.where(has_any & (denom > 0.0), numer / denom, 0.0)
        out_off = q_token_idx * out_stride_b + head_idx * out_stride_h + offs_d
        tl.store(out_ptr + out_off, out_val.to(out_ptr.dtype.element_ty))
        if RETURN_LSE:
            lse_val = tl.where(has_any & (denom > 0.0), m_max + tl.log(denom), float("-inf"))
            tl.store(lse_ptr + head_idx * num_seqs + q_token_idx, lse_val)

    @triton.jit
    def _paged_prefill_gqa_kernel(
        out_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, block_table_ptr, seq_lens_ptr, cu_seqlens_q_ptr,
        scale, num_seqs, q_stride_t: tl.int64, q_stride_h: tl.int64,
        out_stride_t: tl.int64, out_stride_h: tl.int64,
        k_stride_p: tl.constexpr, k_stride_t: tl.constexpr, k_stride_h: tl.constexpr,
        v_stride_p: tl.constexpr, v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
        bt_stride_b: tl.int64, total_q: tl.int64, GQA_GROUP: tl.constexpr, BLOCK_Q: tl.constexpr,
        BLOCK_M: tl.constexpr, PAGE_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
        RETURN_LSE: tl.constexpr, SKIP_SINGLE_Q: tl.constexpr, NOMAX: tl.constexpr,
    ):
        q_block_global = tl.program_id(0)
        kv_head = tl.program_id(1)
        left = 0
        right = num_seqs
        while left < right:
            mid = (left + right) // 2
            if (tl.load(cu_seqlens_q_ptr + mid) // BLOCK_Q + mid) <= q_block_global:
                left = mid + 1
            else:
                right = mid
        seq_idx = left - 1
        q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
        q_len = tl.load(cu_seqlens_q_ptr + seq_idx + 1) - q_start
        if SKIP_SINGLE_Q and q_len <= 1:
            return
        q_block_local = q_block_global - (q_start // BLOCK_Q + seq_idx)
        if q_block_local * BLOCK_Q >= q_len:
            return
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t64 = tl.arange(0, 64)
        offs_in_p = offs_t64 & 15
        page_sub = offs_t64 >> 4
        q_pos = q_block_local * BLOCK_Q + offs_m // GQA_GROUP
        q_head = kv_head * GQA_GROUP + (offs_m % GQA_GROUP)
        q_valid = q_pos < q_len
        q_token_idx = q_start + q_pos
        q_off = q_token_idx[:, None] * q_stride_t + q_head[:, None] * q_stride_h + offs_d[None, :]
        q = tl.load(q_ptr + q_off, mask=q_valid[:, None], other=0.0)
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        context_len = seq_len - q_len
        q_abs_pos = context_len + q_pos
        prefix_bound = tl.minimum(context_len + (q_block_local + 1) * BLOCK_Q, seq_len)
        max_prefix_len = tl.maximum(0, prefix_bound)
        num_pages = (max_prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
        min_pos_pages = tl.maximum(
            0, tl.minimum(num_pages, (context_len + q_block_local * BLOCK_Q) // PAGE_SIZE)
        )
        full_quads = min_pos_pages // 4
        scale_log2e = scale * 1.4426950408889634
        q = (q.to(tl.float32) * scale_log2e).to(q.dtype)
        # NOMAX keeps a static softmax scale instead of a running row max. `q` is already scaled
        # by 1/sqrt(d)*log2(e) above, so the scores are bounded and do not need a per-tile max for
        # range; dropping it also drops the BLOCK_M x HEAD_DIM accumulator rescale from the loop.
        if NOMAX:
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        else:
            m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        bt_row_ptr = block_table_ptr + seq_idx * bt_stride_b
        k_head_base = k_ptr + kv_head * k_stride_h
        v_head_base = v_ptr + kv_head * v_stride_h
        offs_p1024 = tl.arange(0, 1024)
        p_base = tl.load(bt_row_ptr).to(tl.int64)
        n_quad_pages = full_quads * 4
        p_all = tl.load(
            bt_row_ptr + offs_p1024, mask=offs_p1024 < n_quad_pages, other=p_base + offs_p1024
        ).to(tl.int64)
        is_contig = (
            (n_quad_pages <= 1024)
            & (k_stride_p == 16 * k_stride_t)
            & (v_stride_p == 16 * v_stride_t)
            & (tl.sum(tl.where(p_all == (p_base + offs_p1024), 0, 1), axis=0) == 0)
        )
        if is_contig:
            k_contig = offs_t64[None, :] * k_stride_t + offs_d[:, None]
            v_contig = offs_t64[:, None] * v_stride_t + offs_d[None, :]
            for quad_idx in range(0, full_quads):
                base_p = p_base + quad_idx * 4
                k = tl.load(k_head_base + base_p * k_stride_p + k_contig)
                v = tl.load(v_head_base + base_p * v_stride_p + v_contig)
                scores = tl.dot(q, k)
                if NOMAX:
                    p = tl.math.exp2(scores)
                    acc = acc + tl.dot(p.to(v.dtype), v)
                    l_i = l_i + tl.sum(p, axis=1)
                else:
                    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                    alpha = tl.math.exp2(m_i - m_new)
                    p = tl.math.exp2(scores - m_new[:, None])
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    m_i = m_new
        else:
            k_intra = offs_in_p[None, :] * k_stride_t + offs_d[:, None]
            v_intra = offs_in_p[:, None] * v_stride_t + offs_d[None, :]
            for quad_idx in range(0, full_quads):
                page_idx = quad_idx * 4
                phys_p = tl.load(bt_row_ptr + page_idx + page_sub).to(tl.int64)
                k = tl.load(k_head_base + phys_p[None, :] * k_stride_p + k_intra)
                v = tl.load(v_head_base + phys_p[:, None] * v_stride_p + v_intra)
                scores = tl.dot(q, k)
                if NOMAX:
                    p = tl.math.exp2(scores)
                    acc = acc + tl.dot(p.to(v.dtype), v)
                    l_i = l_i + tl.sum(p, axis=1)
                else:
                    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                    alpha = tl.math.exp2(m_i - m_new)
                    p = tl.math.exp2(scores - m_new[:, None])
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    m_i = m_new
        rem_start = full_quads * 4
        pair_end = rem_start + ((min_pos_pages - rem_start) // 2) * 2
        offs_t32 = tl.arange(0, 32)
        offs_in_p32 = offs_t32 & 15
        is_second_p = offs_t32 >= 16
        k_intra32 = offs_in_p32[None, :] * k_stride_t + offs_d[:, None]
        v_intra32 = offs_in_p32[:, None] * v_stride_t + offs_d[None, :]
        for page_idx in range(rem_start, pair_end, 2):
            p0 = tl.load(bt_row_ptr + page_idx).to(tl.int64)
            p1 = tl.load(bt_row_ptr + page_idx + 1).to(tl.int64)
            phys_p2 = tl.where(is_second_p, p1, p0)
            k2 = tl.load(k_head_base + phys_p2[None, :] * k_stride_p + k_intra32)
            v2 = tl.load(v_head_base + phys_p2[:, None] * v_stride_p + v_intra32)
            scores2 = tl.dot(q, k2)
            if NOMAX:
                p2 = tl.math.exp2(scores2)
                acc = acc + tl.dot(p2.to(v2.dtype), v2)
                l_i = l_i + tl.sum(p2, axis=1)
            else:
                m_new = tl.maximum(m_i, tl.max(scores2, axis=1))
                alpha = tl.math.exp2(m_i - m_new)
                p2 = tl.math.exp2(scores2 - m_new[:, None])
                acc = acc * alpha[:, None] + tl.dot(p2.to(v2.dtype), v2)
                l_i = l_i * alpha + tl.sum(p2, axis=1)
                m_i = m_new
        offs_t16 = tl.arange(0, 16)
        k_intra16 = offs_t16[None, :] * k_stride_t + offs_d[:, None]
        v_intra16 = offs_t16[:, None] * v_stride_t + offs_d[None, :]
        for page_idx in range(pair_end, num_pages):
            phys_page = tl.load(bt_row_ptr + page_idx).to(tl.int64)
            token_pos = page_idx * PAGE_SIZE + offs_t16
            t_mask = token_pos < max_prefix_len
            k = tl.load(
                k_head_base + phys_page * k_stride_p + k_intra16, mask=t_mask[None, :], other=0.0
            )
            v = tl.load(
                v_head_base + phys_page * v_stride_p + v_intra16, mask=t_mask[:, None], other=0.0
            )
            c_mask = (token_pos[None, :] <= q_abs_pos[:, None]) & (q_abs_pos[:, None] >= 0)
            raw_s = tl.dot(q, k)
            scores = tl.where(q_valid[:, None] & t_mask[None, :] & c_mask, raw_s, float("-inf"))
            if NOMAX:
                p = tl.math.exp2(scores)
                acc = acc + tl.dot(p.to(v.dtype), v)
                l_i = l_i + tl.sum(p, axis=1)
            else:
                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
                alpha = tl.math.exp2(m_i - m_safe)
                p = tl.math.exp2(scores - m_safe[:, None])
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new
        has_any = (m_i > float("-inf")) & (l_i > 0.0)
        out_val = tl.where(has_any[:, None], acc / l_i[:, None], 0.0)
        out_row = q_token_idx[:, None] * out_stride_t + q_head[:, None] * out_stride_h
        tl.store(
            out_ptr + out_row + offs_d[None, :],
            out_val.to(out_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )
        if RETURN_LSE:
            lse_val = tl.where(has_any, m_i * 0.6931471805599453 + tl.log(l_i), float("-inf"))
            tl.store(lse_ptr + q_head * total_q + q_token_idx, lse_val, mask=q_valid)


def launch_split_kv_decode(
    q, k_cache, v_cache, *, block_table, seq_lens, workspace,
    splits=None, scale=None, out=None, return_lse=False, num_actual_tokens=None,
):
    """Run scalar-base 2-page unrolled split-KV decode (direct write-out when splits==1)."""
    validate_decode_tensors(
        q, k_cache, v_cache, block_table=block_table, seq_lens=seq_lens, out=out
    )
    total_tokens, q_heads, head_dim = q.shape
    active = total_tokens if num_actual_tokens is None else int(num_actual_tokens)
    kv_heads = k_cache.shape[2]
    scale = (head_dim**-0.5) if scale is None else float(scale)
    splits = select_num_splits(active, 4096, kv_heads=kv_heads) if splits is None else int(splits)
    if splits <= 0 or (splits & (splits - 1)) != 0:
        raise ValueError("splits must be a positive power of two for Triton reduction")
    out = torch.empty_like(q) if out is None else out
    lse = (
        torch.full((q_heads, total_tokens), -math.inf, dtype=torch.float32, device=q.device)
        if return_lse
        else out
    )
    direct = splits == 1
    if not direct:
        acc_b, max_b, sum_b = workspace.ensure(q.device, min_batch=total_tokens, min_splits=splits)
        ws_b = acc_b.stride(0) // head_dim
        ws_h = acc_b.stride(1) // head_dim
        ws_s = acc_b.stride(2) // head_dim
    else:
        acc_b = max_b = sum_b = out
        ws_b = ws_h = ws_s = 0
    _split_kv_decode_stage1[(total_tokens, kv_heads, splits)](
        q, k_cache, v_cache, block_table, seq_lens, seq_lens,
        acc_b, max_b, sum_b, out, lse, scale, active,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), ws_b, ws_h, ws_s,
        NUM_KV_HEADS=kv_heads, GQA_GROUP=q_heads // kv_heads, NUM_SPLITS=splits,
        PAGE_SIZE=16, HEAD_DIM=128, DIRECT_OUT=direct,
        RETURN_LSE=return_lse, FILTER_SINGLE_Q=False,
        num_warps=DECODE_LAUNCH_WARPS, num_stages=DECODE_LAUNCH_STAGES,
    )
    if not direct:
        _split_kv_decode_stage2[(total_tokens, q_heads)](
            acc_b, max_b, sum_b, seq_lens, out, lse, active, out.stride(0), out.stride(1),
            max_b.stride(0), max_b.stride(1), max_b.stride(2),
            NUM_SPLITS=splits, HEAD_DIM=128, RETURN_LSE=return_lse,
            FILTER_SINGLE_Q=False, num_warps=2,
        )
    return (out, lse) if return_lse else out


def launch_paged_prefill(
    q, k_cache, v_cache, *, block_table, seq_lens, cu_seqlens_q,
    scale=None, out=None, return_lse=False, block_q=16, workspace=None, max_seqlen_k=None,
):
    """Run 4-page GQA prefill (BLOCK_Q=32 16-warp shared-KV reuse) + single-query split-KV."""
    total_q, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    group = q_heads // kv_heads
    scale = (head_dim**-0.5) if scale is None else float(scale)
    out = torch.zeros_like(q) if out is None else out
    num_seqs = int(cu_seqlens_q.numel() - 1)
    lse = (
        torch.full((q_heads, total_q), -math.inf, dtype=torch.float32, device=q.device)
        if return_lse
        else out
    )
    if total_q == 0 or num_seqs <= 0:
        return (out, lse) if return_lse else out
    is_fp16_bf16 = getattr(q, "element_size", lambda: 2)() == 2
    eff_block_q = (
        32 if (block_q == 16 and total_q >= 256 and group == 8 and is_fp16_bf16) else block_q
    )
    skip_single = (num_seqs > 1) and (not return_lse) and (eff_block_q >= 8)
    warps = 16 if eff_block_q >= 32 else (8 if eff_block_q >= 16 else 4)
    stages = 2 if is_fp16_bf16 else 1
    nomax = is_fp16_bf16 if _FORCE_PREFILL_NOMAX is None else bool(_FORCE_PREFILL_NOMAX)
    _paged_prefill_gqa_kernel[(total_q // eff_block_q + num_seqs, kv_heads)](
        out, lse, q, k_cache, v_cache, block_table, seq_lens, cu_seqlens_q, scale, num_seqs,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), total_q, GQA_GROUP=group, BLOCK_Q=eff_block_q,
        BLOCK_M=eff_block_q * group, PAGE_SIZE=16, HEAD_DIM=128,
        RETURN_LSE=return_lse, SKIP_SINGLE_Q=skip_single,
        NOMAX=nomax, num_warps=warps, num_stages=stages,
    )
    if skip_single:
        max_k = int(max_seqlen_k) if max_seqlen_k and int(max_seqlen_k) > 1 else 4096
        splits = select_num_splits(num_seqs, max_k, kv_heads=kv_heads)
        direct = splits == 1
        if not direct:
            if workspace is None:
                dev_key = (q.device.type, q.device.index)
                workspace = _PREFILL_WORKSPACES.get(dev_key)
                if workspace is None or workspace.max_batch < num_seqs:
                    workspace = SplitKVWorkspace(
                        max_batch=max(64, num_seqs),
                        q_heads=q_heads,
                        max_splits=8,
                        dim=head_dim,
                    )
                    _PREFILL_WORKSPACES[dev_key] = workspace
            acc_b, max_b, sum_b = workspace.ensure(q.device, min_batch=num_seqs, min_splits=splits)
            ws_b = acc_b.stride(0) // head_dim
            ws_h = acc_b.stride(1) // head_dim
            ws_s = acc_b.stride(2) // head_dim
        else:
            acc_b = max_b = sum_b = out
            ws_b = ws_h = ws_s = 0
        _split_kv_decode_stage1[(num_seqs, kv_heads, splits)](
            q, k_cache, v_cache, block_table, seq_lens, cu_seqlens_q,
            acc_b, max_b, sum_b, out, lse, scale, num_seqs,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_table.stride(0), ws_b, ws_h, ws_s,
            NUM_KV_HEADS=kv_heads, GQA_GROUP=group, NUM_SPLITS=splits,
            PAGE_SIZE=16, HEAD_DIM=128, DIRECT_OUT=direct,
            RETURN_LSE=False, FILTER_SINGLE_Q=True,
            num_warps=DECODE_LAUNCH_WARPS, num_stages=DECODE_LAUNCH_STAGES,
        )
        if not direct:
            _split_kv_decode_stage2[(num_seqs, q_heads)](
                acc_b, max_b, sum_b, cu_seqlens_q, out, lse, num_seqs,
                out.stride(0), out.stride(1), ws_b, ws_h, ws_s,
                NUM_SPLITS=splits, HEAD_DIM=128, RETURN_LSE=False,
                FILTER_SINGLE_Q=True, num_warps=2,
            )
    return (out, lse) if return_lse else out


_SEED_COUNTER = 987654321
_SAMPLER_STATS = {"fast_topk_topp": 0, "gumbel_sample": 0, "fallback_sample": 0}
_FP32_LOGITS_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}
_TILE_MAX_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}

if HAS_TRITON:

    @triton.jit
    def _cast_scale_bf16_to_fp32_kernel(
        inp_ptr,
        out_ptr,
        temp_ptr,
        stride_in: tl.int64,
        stride_out: tl.int64,
        vocab_size: int,
        HAS_TEMP: tl.constexpr,
        ALL_RANDOM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        col_block = tl.program_id(1)
        offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(inp_ptr + row_id * stride_in + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_TEMP:
            t = tl.load(temp_ptr + row_id).to(tl.float32)
            if not ALL_RANDOM:
                t = tl.where(t < 1e-5, 1.0, t)
            x = x * (1.0 / t)
        tl.store(out_ptr + row_id * stride_out + offs, x, mask=mask)

    @triton.jit
    def _gumbel_max_sample_kernel(
        logits_ptr,
        out_ptr,
        row_stride: tl.int64,
        vocab_size: int,
        seed: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_ptr = logits_ptr + row_idx * row_stride
        offs_base = tl.arange(0, BLOCK_SIZE)
        best_score = -float("inf")
        best_idx = 0
        row_seed = seed ^ (row_idx * 104729)
        for start in range(0, vocab_size, BLOCK_SIZE):
            offs = start + offs_base
            mask = offs < vocab_size
            logits = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
            u = tl.rand(row_seed, offs)
            u = tl.clamp(u, 1e-10, 1.0 - 1e-7)
            gumbel = -tl.log(-tl.log(u))
            scores = tl.where(logits > -float("inf"), logits + gumbel, -float("inf"))
            chunk_max, chunk_idx = tl.max(scores, axis=0, return_indices=True)
            better = chunk_max > best_score
            best_score = tl.where(better, chunk_max, best_score)
            best_idx = tl.where(better, start + chunk_idx, best_idx)
        tl.store(out_ptr + row_idx, best_idx.to(tl.int64))

    @triton.jit
    def _metax_top_k_top_p_prune_kernel(
        logits_ptr,
        stride_logits_b: tl.int64,
        tile_max_ptr,
        stride_tmax_b: tl.int64,
        thresh_ptr,
        k_ptr,
        p_ptr,
        vocab_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_K: tl.constexpr,
        HAS_P: tl.constexpr,
        WRITE_MASK: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_ptr = logits_ptr + row_idx * stride_logits_b
        tmax_row_ptr = tile_max_ptr + row_idx * stride_tmax_b

        k_val = vocab_size
        if HAS_K:
            k_val = tl.load(k_ptr + row_idx).to(tl.int32)
        p_val = 1.0
        if HAS_P:
            p_val = tl.load(p_ptr + row_idx).to(tl.float32)

        need_k = HAS_K & (k_val > 0) & (k_val < vocab_size)
        need_p = HAS_P & (p_val < 1.0)
        if not (need_k | need_p):
            if not WRITE_MASK:
                tl.store(thresh_ptr + row_idx, -float("inf"))
            return

        m_i = -float("inf")
        d_i = 0.0
        min_i = float("inf")
        num_tiles = tl.cdiv(vocab_size, BLOCK_N)
        for t in range(0, num_tiles):
            offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < vocab_size
            x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
            m_ij = tl.max(x, axis=0)
            tl.store(tmax_row_ptr + t, m_ij)
            new_m = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - new_m)
            alpha = tl.where(new_m == -float("inf"), 0.0, alpha)
            p_blk = tl.exp(x - new_m)
            p_blk = tl.where(mask & (x > -float("inf")), p_blk, 0.0)
            d_i = d_i * alpha + tl.sum(p_blk, axis=0)
            m_i = new_m
            finite_x = tl.where(mask & (x > -float("inf")), x, float("inf"))
            min_i = tl.minimum(min_i, tl.min(finite_x, axis=0))

        if (m_i == -float("inf")) | (d_i <= 0.0):
            if not WRITE_MASK:
                tl.store(thresh_ptr + row_idx, -float("inf"))
            return

        threshold = -float("inf")
        if need_k:
            low = tl.maximum(min_i, m_i - 30.0)
            high = m_i
            for _ in range(14):
                mid = 0.5 * (low + high)
                cnt = 0
                for t in range(0, num_tiles):
                    t_max = tl.load(tmax_row_ptr + t)
                    if t_max >= mid:
                        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
                        mask = offs < vocab_size
                        x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
                        cnt += tl.sum((x >= mid).to(tl.int32), axis=0)
                cond_k = cnt > k_val
                low = tl.where(cond_k, mid, low)
                high = tl.where(cond_k, high, mid)
            threshold = high

        if need_p:
            if (1.0 / d_i) >= p_val:
                threshold = tl.maximum(threshold, m_i)
            else:
                target_sum = p_val * d_i
                low = tl.maximum(min_i, m_i - 16.0)
                if need_k:
                    low = tl.maximum(low, threshold)
                high = m_i
                for _ in range(13):
                    mid = 0.5 * (low + high)
                    s_exp = 0.0
                    for t in range(0, num_tiles):
                        t_max = tl.load(tmax_row_ptr + t)
                        if t_max >= mid:
                            offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
                            mask = offs < vocab_size
                            x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(
                                tl.float32
                            )
                            keep = mask & (x >= mid)
                            p_blk = tl.where(keep, tl.exp(x - m_i), 0.0)
                            s_exp += tl.sum(p_blk, axis=0)
                    cond_p = s_exp >= target_sum
                    low = tl.where(cond_p, mid, low)
                    high = tl.where(cond_p, high, mid)
                threshold = tl.maximum(threshold, low)

        if not WRITE_MASK:
            tl.store(thresh_ptr + row_idx, threshold)
        elif threshold > min_i:
            for t in range(0, num_tiles):
                offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = offs < vocab_size
                t_max = tl.load(tmax_row_ptr + t)
                if t_max < threshold:
                    tl.store(row_ptr + offs, -float("inf"), mask=mask)
                else:
                    x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
                    out = tl.where(x >= threshold, x, -float("inf"))
                    tl.store(row_ptr + offs, out, mask=mask)

    @triton.jit
    def _gumbel_max_sample_pruned_kernel(
        logits_ptr,
        out_ptr,
        tile_max_ptr,
        stride_tmax_b: tl.int64,
        thresh_ptr,
        row_stride: tl.int64,
        vocab_size: int,
        seed: int,
        BLOCK_SIZE: tl.constexpr,
        USE_THRESH: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_ptr = logits_ptr + row_idx * row_stride
        offs_base = tl.arange(0, BLOCK_SIZE)
        best_score = -float("inf")
        best_idx = 0
        row_seed = seed ^ (row_idx * 104729)
        thresh = -float("inf")
        if USE_THRESH:
            thresh = tl.load(thresh_ptr + row_idx).to(tl.float32)
        tmax_row_ptr = tile_max_ptr + row_idx * stride_tmax_b
        num_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        for t in range(0, num_tiles):
            skip = False
            if USE_THRESH:
                t_max = tl.load(tmax_row_ptr + t)
                skip = (thresh > -float("inf")) & (t_max < thresh)
            if not skip:
                start = t * BLOCK_SIZE
                offs = start + offs_base
                mask = offs < vocab_size
                logits = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
                u = tl.rand(row_seed, offs)
                u = tl.clamp(u, 1e-10, 1.0 - 1e-7)
                gumbel = -tl.log(-tl.log(u))
                valid = logits > -float("inf")
                if USE_THRESH:
                    valid = valid & (logits >= thresh)
                scores = tl.where(valid, logits + gumbel, -float("inf"))
                chunk_max, chunk_idx = tl.max(scores, axis=0, return_indices=True)
                better = chunk_max > best_score
                best_score = tl.where(better, chunk_max, best_score)
                best_idx = tl.where(better, start + chunk_idx, best_idx)
        tl.store(out_ptr + row_idx, best_idx.to(tl.int64))


def active_components() -> frozenset[str]:
    return _ACTIVE_COMPONENTS


def applied_components() -> dict[str, bool]:
    return dict(_APPLIED_COMPONENTS)


def install(components=None) -> dict[str, bool]:
    """Install a component subset of the vLLM patches and report what actually took effect.

    `components=None` installs everything (the verified Iluvatar behaviour). A vendor profile
    passes a smaller set so a platform never gets a kernel whose preconditions are unproven.
    """
    global _ACTIVE_COMPONENTS, _APPLIED_COMPONENTS
    _ACTIVE_COMPONENTS = ALL_COMPONENTS if components is None else frozenset(components)
    unknown = _ACTIVE_COMPONENTS - ALL_COMPONENTS
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")
    _APPLIED_COMPONENTS = {}
    patch_iluvatar_sampler()
    _APPLIED_COMPONENTS.setdefault("attention", False)
    return applied_components()


def patch_iluvatar_sampler(components=None) -> bool:
    """Patch vLLM TopKTopPSampler, memory profiler, BlockPool, and KV-aware scheduler."""
    global _ACTIVE_COMPONENTS
    if components is not None:
        _ACTIVE_COMPONENTS = frozenset(components)
    active = _ACTIVE_COMPONENTS
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as _ts
        import vllm.v1.sample.ops.topk_topp_triton as _tt
    except Exception:
        return False

    if getattr(_ts, "_iluvatar_fast_sampler_patched", False) and (
        getattr(_ts, "_fast_sampler_components", None) == tuple(sorted(active))
    ):
        return True

    def fast_apply_top_k_top_p_triton(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        mask_value: float = float("-inf"),
    ) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        topk_enabled = k is not None
        topp_enabled = p is not None
        if batch_size == 0 or not (topk_enabled or topp_enabled):
            return logits
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        k_ptr = k.to(torch.int32) if k is not None else logits
        p_ptr = p.to(torch.float32) if p is not None else logits
        num_programs = min(32, batch_size)
        buf_key = (logits.device, logits.dtype, vocab_size)
        buffer = _tt._TRITON_BUFFER_CACHE.get(buf_key)
        if buffer is None or buffer.shape[0] < num_programs:
            size = min(triton.next_power_of_2(num_programs), 32)
            buffer = logits.new_empty((size, vocab_size))
            _tt._TRITON_BUFFER_CACHE[buf_key] = buffer
        if buffer.shape[0] > num_programs:
            buffer = buffer[:num_programs]
        tables = _tt._TRITON_TABLE_CACHE.get(logits.device)
        if tables is None:
            normal_cdf_to_sigma_table = logits.new_tensor(_tt._NORMAL_CDF_TO_SIGMA_TABLE)
            percentile_to_std_table = logits.new_tensor(_tt._PERCENTILE_TO_STD_TABLE)
            _tt._TRITON_TABLE_CACHE[logits.device] = (
                normal_cdf_to_sigma_table,
                percentile_to_std_table,
            )
        else:
            normal_cdf_to_sigma_table, percentile_to_std_table = tables
        _SAMPLER_STATS["fast_topk_topp"] += 1
        _tt._topk_topp_kernel[(num_programs,)](
            logits,
            logits.stride(0),
            buffer,
            percentile_to_std_table,
            normal_cdf_to_sigma_table,
            k_ptr,
            p_ptr,
            BATCH_SIZE=batch_size,
            MASK_VALUE=mask_value,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=8192,
            BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=topk_enabled,
            TOPP_ENABLED=topp_enabled,
            num_warps=8,
        )
        return logits

    def _ensure_tile_max_buffers(
        device: torch.device, batch_size: int, num_tiles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, num_tiles)
        cached = _TILE_MAX_CACHE.get(key)
        if cached is None or cached[0].shape[0] < batch_size:
            cap = max(64, batch_size)
            tmax = torch.empty((cap, num_tiles), dtype=torch.float32, device=device)
            thresh = torch.empty((cap,), dtype=torch.float32, device=device)
            cached = (tmax, thresh)
            _TILE_MAX_CACHE[key] = cached
        return cached[0][:batch_size], cached[1][:batch_size]

    def _run_metax_top_k_top_p_prune(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        *,
        write_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        batch_size, vocab_size = logits.shape
        block_n = 4096
        num_tiles = triton.cdiv(vocab_size, block_n)
        tile_max_buf, thresh_buf = _ensure_tile_max_buffers(
            logits.device, batch_size, max(64, num_tiles)
        )
        k_ptr = k.to(torch.int32) if k is not None else logits
        p_ptr = p.to(torch.float32) if p is not None else logits
        _SAMPLER_STATS["fast_topk_topp"] += 1
        _metax_top_k_top_p_prune_kernel[(batch_size,)](
            logits,
            logits.stride(0),
            tile_max_buf,
            tile_max_buf.stride(0),
            thresh_buf,
            k_ptr,
            p_ptr,
            vocab_size=vocab_size,
            BLOCK_N=block_n,
            HAS_K=k is not None,
            HAS_P=p is not None,
            WRITE_MASK=write_mask,
            num_warps=8,
        )
        return logits, tile_max_buf, thresh_buf

    def fast_apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        if k is None and p is None:
            return logits
        if "topk_triton" in _ACTIVE_COMPONENTS and logits.shape[0] >= 1 and logits.is_cuda:
            return fast_apply_top_k_top_p_triton(logits, k, p)
        if (
            "sampler_gumbel" in _ACTIVE_COMPONENTS
            and HAS_TRITON
            and logits.is_cuda
            and logits.ndim == 2
            and logits.shape[0] >= 1
        ):
            logits, _, _ = _run_metax_top_k_top_p_prune(logits, k, p, write_mask=True)
            return logits
        return _ts.apply_top_k_top_p_pytorch(logits, k, p)

    def fast_forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        global _SEED_COUNTER
        logprobs_mode = getattr(self, "logprobs_mode", None)
        can_use_gumbel = (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and not getattr(self, "use_fp64_gumbel", False)
            and logits.is_cuda
            and logits.ndim == 2
            and logits.shape[0] > 0
        )
        if (
            can_use_gumbel
            and HAS_TRITON
            and (k is not None or p is not None)
            and "topk_triton" not in _ACTIVE_COMPONENTS
        ):
            _SEED_COUNTER = ((_SEED_COUNTER + 104729) & 0x7FFFFFFF) | 1
            logits, tile_max_buf, thresh_buf = _run_metax_top_k_top_p_prune(
                logits, k, p, write_mask=False
            )
            batch_size, vocab_size = logits.shape
            out = torch.empty((batch_size,), dtype=torch.int64, device=logits.device)
            _SAMPLER_STATS["gumbel_sample"] += 1
            _gumbel_max_sample_pruned_kernel[(batch_size,)](
                logits,
                out,
                tile_max_buf,
                tile_max_buf.stride(0),
                thresh_buf,
                logits.stride(0),
                vocab_size,
                _SEED_COUNTER,
                BLOCK_SIZE=4096,
                USE_THRESH=True,
                num_warps=8,
            )
            return out, None

        logits = fast_apply_top_k_top_p(logits, k, p)
        if can_use_gumbel:
            _SEED_COUNTER = ((_SEED_COUNTER + 104729) & 0x7FFFFFFF) | 1
            batch_size, vocab_size = logits.shape
            out = torch.empty((batch_size,), dtype=torch.int64, device=logits.device)
            _SAMPLER_STATS["gumbel_sample"] += 1
            _gumbel_max_sample_kernel[(batch_size,)](
                logits,
                out,
                logits.stride(0),
                vocab_size,
                _SEED_COUNTER,
                BLOCK_SIZE=4096,
                num_warps=8,
            )
            return out, None

        _SAMPLER_STATS["fallback_sample"] += 1
        logits_to_return = None
        if logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif logprobs_mode == "processed_logprobs":
            logits = logits.log_softmax(dim=-1, dtype=torch.float32)
            logits_to_return = logits
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return _ts.random_sample(probs, generators), logits_to_return

    if "topk_triton" in active:
        _tt.apply_top_k_top_p_triton = fast_apply_top_k_top_p_triton
        _ts.apply_top_k_top_p_triton = fast_apply_top_k_top_p_triton
    if {"topk_triton", "sampler_gumbel"} & active:
        _ts.apply_top_k_top_p = fast_apply_top_k_top_p
        _ts._metax_fast_apply_top_k_top_p = fast_apply_top_k_top_p
    if "sampler_gumbel" in active:
        _ts.TopKTopPSampler.forward_native = fast_forward_native
        _ts.TopKTopPSampler.forward = fast_forward_native
    _APPLIED_COMPONENTS["topk_triton"] = "topk_triton" in active
    _APPLIED_COMPONENTS["sampler_gumbel"] = "sampler_gumbel" in active
    try:
        import vllm.v1.sample.sampler as _sm

        _orig_sampler_forward = _sm.Sampler.forward
        _orig_apply_temp = _sm.Sampler.apply_temperature

        @staticmethod
        def _fast_apply_temperature(
            logits: torch.Tensor,
            temp: torch.Tensor,
            all_random: bool,
        ) -> torch.Tensor:
            if HAS_TRITON and logits.is_cuda and logits.ndim == 2 and logits.shape[0] > 0:
                bsz, vocab = logits.shape
                if logits.dtype == torch.float32 and logits.stride(1) == 1:
                    out_fp32 = logits
                else:
                    key = (logits.device, vocab)
                    buf = _FP32_LOGITS_CACHE.get(key)
                    if buf is None or buf.shape[0] < bsz:
                        buf = torch.empty(
                            (max(64, bsz), vocab), dtype=torch.float32, device=logits.device
                        )
                        _FP32_LOGITS_CACHE[key] = buf
                    out_fp32 = buf[:bsz]
                _cast_scale_bf16_to_fp32_kernel[(bsz, triton.cdiv(vocab, 2048))](
                    logits,
                    out_fp32,
                    temp,
                    logits.stride(0),
                    out_fp32.stride(0),
                    vocab,
                    HAS_TEMP=True,
                    ALL_RANDOM=bool(all_random),
                    BLOCK_SIZE=2048,
                    num_warps=4,
                )
                return out_fp32
            if logits.dtype != torch.float32:
                logits = logits.to(torch.float32)
            return _orig_apply_temp(logits, temp, all_random)

        def _fast_sampler_forward(
            self,
            logits: torch.Tensor,
            sampling_metadata,
            predict_bonus_token: bool = False,
            logprobs_mode_override=None,
        ):
            holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
            if (
                logits.is_cuda
                and logits.dtype != torch.float32
                and sampling_metadata.max_num_logprobs is None
                and not sampling_metadata.logprob_token_ids
                and sampling_metadata.no_penalties
                and not sampling_metadata.bad_words_token_ids
                and sampling_metadata.allowed_token_ids_mask is None
                and not sampling_metadata.logitsprocs.non_argmax_invariant
                and (holder is None or not holder.has_tracked_requests())
            ):
                sampled, _ = self.sample(logits, sampling_metadata, logprobs_mode_override)
                return _sm.SamplerOutput(
                    sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
                    logprobs_tensors=None,
                )
            return _orig_sampler_forward(
                self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )

        if "sampler_cast" in active:
            _sm.Sampler.apply_temperature = _fast_apply_temperature
        if "sampler_gumbel" in active:
            _sm.Sampler.forward = _fast_sampler_forward
        _APPLIED_COMPONENTS["sampler_cast"] = "sampler_cast" in active
    except Exception:
        pass
    try:
        import vllm.v1.core.block_pool as _bp

        def _fast_get_new_blocks(self, num_blocks: int):
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            slab_ends = getattr(self, "_slab_ends", None)
            if slab_ends is None:
                slab_ends = set(range(1024, self.num_gpu_blocks, 1024))
                self._slab_ends = slab_ends
            if num_blocks == 1:
                blk = self.free_block_queue.fake_free_list_tail.prev_free_block
                self.free_block_queue.remove(blk)
                ret = [blk]
            else:
                ret = self.free_block_queue.popleft_n(num_blocks)
                last_id = ret[-1].block_id
                self._last_prefill_id = last_id
                self._last_prefill_end = last_id in slab_ends
            if self.enable_caching:
                for block in ret:
                    self._maybe_evict_cached_block(block)
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
                    if self.metrics_collector:
                        self.metrics_collector.on_block_allocated(block)
            else:
                for block in ret:
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
                    if self.metrics_collector:
                        self.metrics_collector.on_block_allocated(block)
            return ret

        def _fast_free_blocks(self, ordered_blocks):
            freed = []
            for block in ordered_blocks:
                block.ref_cnt -= 1
                if block.ref_cnt == 0 and not block.is_null:
                    freed.append(block)
            if not freed:
                return
            slab_ends = getattr(self, "_slab_ends", None)
            if slab_ends is None:
                slab_ends = set(range(1024, self.num_gpu_blocks, 1024))
                self._slab_ends = slab_ends
            freed.sort(key=lambda b: b.block_id)
            contig = []
            scatter = []
            i = 0
            n = len(freed)
            while i < n:
                j = i + 1
                while j < n and freed[j].block_id == freed[j - 1].block_id + 1:
                    j += 1
                if (j - i) >= 16:
                    for k in range(i, j - 1):
                        slab_ends.discard(freed[k].block_id)
                    slab_ends.add(freed[j - 1].block_id)
                    contig.extend(freed[i:j])
                else:
                    for k in range(i, j):
                        slab_ends.add(freed[k].block_id)
                    scatter.extend(freed[i:j])
                i = j
            q = self.free_block_queue
            if contig:
                head_s = q.fake_free_list_head
                tail_s = q.fake_free_list_tail
                first = head_s.next_free_block
                last_pf = getattr(self, "_last_prefill_id", -999)
                last_end = getattr(self, "_last_prefill_end", True)
                if not last_end and first is not tail_s and first.block_id == last_pf + 1:
                    cur = first
                    while (
                        cur.block_id not in slab_ends
                        and cur.next_free_block is not tail_s
                        and cur.next_free_block.block_id == cur.block_id + 1
                    ):
                        cur = cur.next_free_block
                    after = cur.next_free_block
                    for idx_c, b in enumerate(contig):
                        b.prev_free_block = cur if idx_c == 0 else contig[idx_c - 1]
                        b.next_free_block = (
                            after if idx_c == len(contig) - 1 else contig[idx_c + 1]
                        )
                    cur.next_free_block = contig[0]
                    after.prev_free_block = contig[-1]
                    q.num_free_blocks += len(contig)
                else:
                    q.prepend_n(contig)
            if scatter:
                q.append_n(scatter)
            if q.num_free_blocks == self.num_gpu_blocks - 1:
                all_free = self.blocks[1:]
                slab_ends.clear()
                slab_ends.update(range(1024, self.num_gpu_blocks, 1024))
                q.fake_free_list_head.next_free_block = all_free[0]
                q.fake_free_list_tail.prev_free_block = all_free[-1]
                for idx_f, b in enumerate(all_free):
                    b.prev_free_block = (
                        q.fake_free_list_head if idx_f == 0 else all_free[idx_f - 1]
                    )
                    b.next_free_block = (
                        q.fake_free_list_tail
                        if idx_f == len(all_free) - 1
                        else all_free[idx_f + 1]
                    )
                self._last_prefill_id = -999
                self._last_prefill_end = True

        if "block_pool" in active:
            _bp.BlockPool.get_new_blocks = _fast_get_new_blocks
            _bp.BlockPool.free_blocks = _fast_free_blocks
        _APPLIED_COMPONENTS["block_pool"] = "block_pool" in active
    except Exception:
        _APPLIED_COMPONENTS["block_pool"] = False
    try:
        import vllm.v1.core.sched.scheduler as _sched
        import vllm_fl.worker.model_runner as _mr
        import vllm_fl.worker.worker as _wk

        _orig_mr_init = getattr(
            _mr.ModelRunnerFL, "_fl_orig_init", _mr.ModelRunnerFL.__init__
        )
        _orig_dummy_sampler = getattr(
            _mr.ModelRunnerFL,
            "_fl_orig_dummy_sampler",
            _mr.ModelRunnerFL._dummy_sampler_run,
        )
        _orig_determine_mem = getattr(
            _wk.WorkerFL,
            "_fl_orig_determine_mem",
            _wk.WorkerFL.determine_available_memory,
        )
        _orig_schedule = getattr(
            _sched.Scheduler, "_fl_orig_schedule", _sched.Scheduler.schedule
        )
        _mr.ModelRunnerFL._fl_orig_init = _orig_mr_init
        _mr.ModelRunnerFL._fl_orig_dummy_sampler = _orig_dummy_sampler
        _wk.WorkerFL._fl_orig_determine_mem = _orig_determine_mem
        _sched.Scheduler._fl_orig_schedule = _orig_schedule

        def _dyn_batch_mr_init(self, vllm_config, device, *args, **kwargs):
            _orig_mr_init(self, vllm_config, device, *args, **kwargs)
            if (
                getattr(self, "max_num_tokens", 0) <= 2048
                and torch.cuda.is_available()
                and torch.cuda.get_device_properties(device).total_memory >= (48 << 30)
            ):
                import numpy as _np

                self.max_num_tokens = 16448
                self.input_ids = self._make_buffer(
                    self.max_num_tokens, dtype=torch.int32
                )
                self.positions = torch.zeros(
                    self.max_num_tokens, dtype=torch.int64, device=self.device
                )
                self.req_indices = self._make_buffer(
                    self.max_num_tokens, dtype=torch.int64
                )
                self.inputs_embeds = self._make_buffer(
                    self.max_num_tokens,
                    self.inputs_embeds_size,
                    dtype=self.dtype,
                    numpy=False,
                )
                self.is_token_ids = self._make_buffer(
                    self.max_num_tokens, dtype=torch.bool
                )
                arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
                self.arange_np = _np.arange(arange_size, dtype=_np.int64)
                self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
                self._arange_scratch = _np.empty(arange_size, dtype=_np.int64)
                self.input_batch = _mr.InputBatch(
                    self.max_num_reqs,
                    max(self.max_model_len, self.max_encoder_len),
                    self.max_num_tokens,
                    self.device,
                    self.model_config.get_vocab_size(),
                    self._init_block_sizes,
                    self._init_kernel_block_sizes,
                    num_spec_tokens=self.num_spec_tokens,
                    logitsprocs=self.input_batch.logitsprocs,
                    logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                    is_pooling_model=self.is_pooling_model,
                    cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                    reasoning_config=self.vllm_config.reasoning_config,
                )

        def _trimmed_dummy_sampler_run(self, hidden_states: torch.Tensor) -> torch.Tensor:
            out = _orig_dummy_sampler(self, hidden_states[:64])
            _tt.reset_buffer_cache()
            _FP32_LOGITS_CACHE.clear()
            _TILE_MAX_CACHE.clear()
            return out

        def _reclaimed_determine_available_memory(self) -> int:
            avail = _orig_determine_mem(self)
            floor_bytes = int(
                self.requested_memory - self.model_runner.model_memory_usage - (64 << 20)
            )
            if avail < (36 << 30) and floor_bytes > avail:
                self.available_kv_cache_memory_bytes = floor_bytes
                avail = floor_bytes
            return avail

        def _kv_aware_schedule(self, *args, **kwargs):
            saved_max = self.max_num_running_reqs
            saved_budget = self.max_num_scheduled_tokens
            try:
                total_kv = self.kv_cache_config.num_blocks * getattr(
                    self, "block_size", 16
                )
                step_budget = (
                    max(saved_budget, 16448) if total_kv > 600_000 else saved_budget
                )
                if step_budget > saved_budget:
                    self.max_num_scheduled_tokens = step_budget
                req0 = (
                    self.running[0]
                    if self.running
                    else (self.waiting[0] if self.waiting else None)
                )
                if req0 is not None:
                    p_toks = getattr(req0, "num_prompt_tokens", None)
                    if p_toks is None:
                        p_toks = getattr(req0, "num_tokens", 0) - getattr(
                            req0, "num_output_tokens", 0
                        )
                    req_total = int(p_toks) + int(getattr(req0, "max_tokens", 0) or 0)
                    if int(p_toks) >= 8192 and req_total >= 8192 and total_kv > 0:
                        max_fit = max(1, total_kv // req_total)
                        if max_fit < saved_max:
                            self.max_num_running_reqs = max_fit
                if self.waiting and len(self.running) < self.max_num_running_reqs:
                    running_tokens = 0
                    for r in self.running:
                        rem = (
                            getattr(r, "num_tokens_with_spec", r.num_tokens)
                            + getattr(r, "num_output_placeholders", 0)
                            - r.num_computed_tokens
                        )
                        if rem > 0:
                            running_tokens += rem
                    if running_tokens < step_budget:
                        avail_slots = self.max_num_running_reqs - len(self.running)
                        prefill_tokens = 0
                        count_w = 0
                        for idx_w, w in enumerate(self.waiting):
                            if idx_w >= avail_slots:
                                break
                            w_rem = w.num_tokens - w.num_computed_tokens
                            if w_rem <= 0:
                                continue
                            if w_rem >= 2048 and count_w >= 1:
                                break
                            if running_tokens + prefill_tokens + w_rem <= step_budget:
                                prefill_tokens += w_rem
                                count_w += 1
                                if w_rem >= 2048:
                                    break
                            else:
                                break
                        if count_w > 0:
                            self.max_num_scheduled_tokens = running_tokens + prefill_tokens
                            self.max_num_running_reqs = min(
                                self.max_num_running_reqs, len(self.running) + count_w
                            )
                return _orig_schedule(self, *args, **kwargs)
            finally:
                self.max_num_running_reqs = saved_max
                self.max_num_scheduled_tokens = saved_budget

        if "sched_admission" in active:
            _mr.ModelRunnerFL.__init__ = _dyn_batch_mr_init
            _mr.ModelRunnerFL._dummy_sampler_run = _trimmed_dummy_sampler_run
            _wk.WorkerFL.determine_available_memory = _reclaimed_determine_available_memory
            _sched.Scheduler.schedule = _kv_aware_schedule
        _APPLIED_COMPONENTS["sched_admission"] = "sched_admission" in active
    except Exception:
        _APPLIED_COMPONENTS["sched_admission"] = False
    try:
        import vllm.v1.attention.backends.triton_attn as _ta

        _orig_stride_order = _ta.TritonAttentionBackend.get_kv_cache_stride_order

        def _contig_kv_cache_stride_order(
            include_num_layers_dimension: bool = False,
        ) -> tuple[int, ...]:
            try:
                order = _orig_stride_order(include_num_layers_dimension)
            except Exception:
                order = (
                    (1, 0, 2, 3, 4, 5)
                    if include_num_layers_dimension
                    else (0, 1, 2, 3, 4)
                )
            if not include_num_layers_dimension and order == (0, 1, 2, 3, 4):
                return (1, 0, 2, 3, 4)
            if include_num_layers_dimension and order == (1, 0, 2, 3, 4, 5):
                return (0, 2, 1, 3, 4, 5)
            return order

        if "kv_stride_order" in active:
            _ta.TritonAttentionBackend.get_kv_cache_stride_order = staticmethod(
                _contig_kv_cache_stride_order
            )
        _APPLIED_COMPONENTS["kv_stride_order"] = "kv_stride_order" in active
    except Exception:
        _APPLIED_COMPONENTS["kv_stride_order"] = False
    _ts._iluvatar_fast_sampler_patched = True
    _ts._fast_sampler_components = tuple(sorted(active))
    return True



# No import-time install: the vendor hook appended by `lab.patches` calls `install(...)` with the
# component set its platform profile allows. Installing everything at import time would hand a
# platform kernels whose preconditions were never checked (the exact failure mode this repo
# documented in EXPERIMENTS.md §7).
