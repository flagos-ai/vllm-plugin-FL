# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU fallbacks for the upstream MiniMax-M3 model.

Upstream ``vllm.models.minimax_m3`` is CUDA/Triton end to end; five per-layer
steps are replaced here: the flashinfer Gemma RMSNorm, the ``torch.ops._C``
qk-norm/RoPE/cache-insert op, the Triton lightning indexer, the four-``tl.dot``
block-sparse attention, and the ROCm SwiGLU-OAI MoE activation.  The
indexer/attend twins are correctness ports, graded against a CPU oracle
(cross-check: transformers' ``minimax_m3_vl``); the flashinfer-free allreduce
rmsnorm sibling needs no patch (it bails out off CUDA).

"""

from __future__ import annotations

import json
import os
import sys

import torch

_applied = False

# Sparse attention block size == paged-KV page size for MiniMax-M3.
_SPARSE_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Shared math (pure torch, mirroring transformers' minimax_m3_vl)
# ---------------------------------------------------------------------------
def _gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gemma-style RMSNorm: normalize in fp32, scale by ``weight`` (already 1+w).

    ``out_dtype`` defaults to the input dtype; the fused op passes fp32 so the
    norm and the rope that follows stay fp32 and the value is rounded once.
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(out_dtype or x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cos_sin(
    cos_sin_cache: torch.Tensor, positions: torch.Tensor, rotary_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand vllm's ``(cos | sin)`` cache rows to full ``rotary_dim`` width.

    ``cos_sin_cache`` rows are ``[cos(rotary_dim/2) | sin(rotary_dim/2)]``;
    rotate-half rope applies the same cos/sin to the first and second half, so
    both are duplicated here.
    """
    row = cos_sin_cache.index_select(0, positions.to(torch.long)).float()
    half = rotary_dim // 2
    cos = torch.cat((row[..., :half], row[..., :half]), dim=-1)
    sin = torch.cat((row[..., half:rotary_dim], row[..., half:rotary_dim]), dim=-1)
    return cos, sin


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Rotate-half rope on the first ``rotary_dim`` channels of ``x[N, H, D]``.

    Computed in fp32 (the oracle runs fp32 end to end); the cache itself decides
    where the rotation stops, the remaining channels pass through untouched.
    """
    out_dtype = out_dtype or x.dtype
    rot, pas = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    cos = cos[:, None, :].to(torch.float32)
    sin = sin[:, None, :].to(torch.float32)
    out = rot * cos + _rotate_half(rot) * sin
    return torch.cat((out, pas.float()), dim=-1).to(out_dtype)


# ---------------------------------------------------------------------------
# 1. Gemma RMSNorm (flashinfer -> pure torch)
# ---------------------------------------------------------------------------
def _forward_lpu(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # ``weight`` is a zeros-initialized fp32 parameter; ``+1`` is the Gemma
    # affine.  Materialize the shifted scale once per call.
    weight = self.weight.float() + 1.0
    if residual is None:
        return _gemma_rms_norm(x, weight, self.variance_epsilon)
    new_residual = x + residual
    residual.copy_(new_residual)
    return _gemma_rms_norm(new_residual, weight, self.variance_epsilon), residual


# ---------------------------------------------------------------------------
# 2. fused qk-norm + rope + paged insert (torch.ops._C -> pure torch)
# ---------------------------------------------------------------------------
def _scatter_slots(dst: torch.Tensor, slots: list[int], src: torch.Tensor,
                   page_size: int, kv_index: int | None = None) -> None:
    """Write ``src[i]`` into flat slot ``slots[i]`` of a paged cache.

    ``dst`` is a paged cache of ``[num_pages, page_size, ...]`` (K/V caches are
    ``[num_pages, 2, page_size, ...]`` with the K-or-V axis selected by
    ``kv_index``).  Contiguous ascending runs are written with a single slice
    copy; scattered slots degrade to one copy per token.  Writes go through
    ``copy_`` into a slice -- the pattern the DSV4/GLM paged helpers use,
    because ``clone`` on a double-die split view is rejected on LPU.
    """
    n = len(slots)
    start = 0
    while start < n:
        page = slots[start] // page_size
        off = slots[start] % page_size
        end = start + 1
        while (
            end < n
            and slots[end] == slots[end - 1] + 1
            and slots[end] // page_size == page
        ):
            end += 1
        run = end - start
        if kv_index is None:
            dst[page, off:off + run].copy_(src[start:end])
        else:
            dst[page, kv_index, off:off + run].copy_(src[start:end])
        start = end


@torch.no_grad()
def _fused_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    """Pure-torch twin of ``torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert``.

    ``qkv`` holds ``[q | k | v | index_q | index_k]`` per token.  q/k/index_q/
    index_k get a per-head Gemma RMSNorm followed by partial rotate-half RoPE;
    v is copied through.  The results go to ``q_out``/``index_q_out`` when the
    caller supplies them (the sparse layer does), and k/v/index_k are scattered
    into the paged caches.  Without ``q_out`` the q/index_q results are written
    back into ``qkv`` in place, which is what the dense (non-sparse) layer does.
    """
    if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise NotImplementedError(
            f"MiniMax-M3 fused qknorm/rope/insert twin: kv_cache_dtype="
            f"{kv_cache_dtype!r} (quantized KV) is not ported to LPU"
        )

    num_tokens = qkv.shape[0]
    head_dim = q_norm_weight.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q = qkv[:, :q_size].view(num_tokens, num_heads, head_dim)
    k = qkv[:, q_size:q_size + kv_size].view(
        num_tokens, num_kv_heads, head_dim)
    off = q_size + 2 * kv_size
    v = qkv[:, q_size + kv_size:off].view(num_tokens, num_kv_heads, head_dim)

    # Norm and rope run in fp32 and are rounded exactly once, when the result
    # is written into its destination (``copy_`` casts).
    cos, sin = _rope_cos_sin(cos_sin_cache, positions, rotary_dim)
    fp32 = torch.float32
    q = _apply_partial_rope(
        _gemma_rms_norm(q, q_norm_weight.float() + 1.0, eps, fp32),
        cos, sin, rotary_dim, fp32)
    k = _apply_partial_rope(
        _gemma_rms_norm(k, k_norm_weight.float() + 1.0, eps, fp32),
        cos, sin, rotary_dim, fp32)

    q_flat = q.reshape(num_tokens, q_size)
    if q_out is not None:
        q_out.copy_(q_flat)
    else:
        qkv[:, :q_size].copy_(q_flat)

    if kv_cache is not None and slot_mapping is not None:
        slots = slot_mapping.tolist()
        page_size = block_size or kv_cache.shape[2]
        _scatter_slots(kv_cache, slots, k, page_size, kv_index=0)
        _scatter_slots(kv_cache, slots, v, page_size, kv_index=1)

    if num_index_heads <= 0:
        return

    idx_dim = index_q_norm_weight.shape[0]
    idx_q_size = num_index_heads * idx_dim
    idx_q = qkv[:, off:off + idx_q_size].view(
        num_tokens, num_index_heads, idx_dim)
    idx_k = qkv[:, off + idx_q_size:off + idx_q_size + idx_dim]

    idx_q = _apply_partial_rope(
        _gemma_rms_norm(idx_q, index_q_norm_weight.float() + 1.0, eps, fp32),
        cos, sin, rotary_dim, fp32)
    # The index key is a single shared head: keep a size-1 head axis so the
    # rope broadcast is per token rather than token-by-token.
    idx_k = _apply_partial_rope(
        _gemma_rms_norm(idx_k.unsqueeze(-2),
                        index_k_norm_weight.float() + 1.0, eps, fp32),
        cos, sin, rotary_dim, fp32).squeeze(-2)

    idx_q_flat = idx_q.reshape(num_tokens, idx_q_size)
    if index_q_out is not None:
        index_q_out.copy_(idx_q_flat.to(index_q_out.dtype))
    else:
        qkv[:, off:off + idx_q_size].copy_(idx_q_flat)

    if index_cache is not None and index_slot_mapping is not None:
        _scatter_slots(index_cache, index_slot_mapping.tolist(), idx_k,
                       index_cache.shape[1])


# ---------------------------------------------------------------------------
# MoE activation (SwiGLU-OAI)
# ---------------------------------------------------------------------------
def _swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Pure-PyTorch twin of ``amd/ops/swiglu_oai.swiglu_oai_split``.

    SwiGLU-OAI on a packed ``[..., 2I]`` tensor (gate = first half, up = second
    half)::

        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, +limit)
        out  = gate * sigmoid(alpha * gate) * (up + beta)

    The intermediates stay in fp32 and only the result rounds, which is what
    the kernel does too -- ROCm keeps this kernel over
    ``_C::silu_and_mul_with_clamp`` precisely because the built op rounds every
    intermediate to bf16 (rel ~3e-3, against ~1e-6 here).
    """
    n_inter = gate_up.shape[-1] // 2
    flat = gate_up.reshape(-1, 2 * n_inter)
    gate = flat[:, :n_inter].float()
    up = flat[:, n_inter:].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = gate * torch.sigmoid(alpha * gate) * (up + beta)
    dtype = out_dtype if out_dtype is not None else gate_up.dtype
    return out.to(dtype).reshape(*gate_up.shape[:-1], n_inter)


def _install_swiglu_oai_twin() -> None:
    """Point the MoE activation kernel at the twin.

    The plugin calls ``swiglu_oai_split`` through a function-local import, so
    rebinding the kernel's own module is enough for it.  The package re-exports
    the name and the ROCm model module imports it eagerly; rebind whatever is
    already loaded so no live caller keeps the Triton version (the LPU runs the
    text/nvidia path, so ``amd.model`` is normally absent -- look it up rather
    than importing it for no reason).
    """
    from vllm.models.minimax_m3.amd import ops as amd_ops
    from vllm.models.minimax_m3.amd.ops import swiglu_oai

    if not hasattr(swiglu_oai, "swiglu_oai_split"):
        raise AttributeError(
            "MiniMax-M3 LPU patch: swiglu_oai.swiglu_oai_split is missing; "
            "upstream moved the MoE activation kernel")
    swiglu_oai.swiglu_oai_split = _swiglu_oai_split
    amd_ops.swiglu_oai_split = _swiglu_oai_split
    amd_model = sys.modules.get("vllm.models.minimax_m3.amd.model")
    if amd_model is not None:
        amd_model.swiglu_oai_split = _swiglu_oai_split


# ---------------------------------------------------------------------------
# Paged-cache reads for the two attention twins
# ---------------------------------------------------------------------------
def _gather_kv_pages(kv_cache: torch.Tensor, pages: list[int], n: int):
    """Copy ``pages`` out of the paged K/V cache into dense ``[n, nkv, hd]``.

    Reads are ``copy_`` from index slices of the (multi-GiB) paged tensor into a
    small dense buffer, the same direction the DSV4/GLM paged helpers use.
    """
    page_size = kv_cache.shape[2]
    nkv, head_dim = kv_cache.shape[3], kv_cache.shape[4]
    k = torch.empty((len(pages) * page_size, nkv, head_dim),
                    dtype=kv_cache.dtype, device=kv_cache.device)
    v = torch.empty_like(k)
    for i, page in enumerate(pages):
        k[i * page_size:(i + 1) * page_size].copy_(kv_cache[page, 0])
        v[i * page_size:(i + 1) * page_size].copy_(kv_cache[page, 1])
    return k[:n], v[:n]


def _gather_index_pages(index_cache: torch.Tensor, pages: list[int],
                        n: int) -> torch.Tensor:
    """Copy the indexer side cache for ``pages`` into a dense ``[n, idx_dim]``."""
    page_size = index_cache.shape[1]
    out = torch.empty((len(pages) * page_size, index_cache.shape[2]),
                      dtype=index_cache.dtype, device=index_cache.device)
    for i, page in enumerate(pages):
        out[i * page_size:(i + 1) * page_size].copy_(index_cache[page])
    return out[:n]


def _num_pages(seq_len: int) -> int:
    return (seq_len + _SPARSE_BLOCK_SIZE - 1) // _SPARSE_BLOCK_SIZE


def _block_keep(sel: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """``[nidx, nq, topk]`` block ids -> ``[nidx, nq, num_blocks]`` boolean.

    Membership is a broadcast comparison rather than a scatter so the twin does
    not depend on advanced-index support; ``-1`` padding never matches a real
    block id.
    """
    blocks = torch.arange(num_blocks, device=sel.device)
    return (sel.unsqueeze(-1) == blocks).any(dim=-2)


_DEBUG_SELECT = os.environ.get("VLLM_FL_M3_DEBUG_SELECT")
_debug_select_calls = 0


def _index_select_blocks(
    idx_q: torch.Tensor,
    k_all: torch.Tensor,
    q_pos: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Per-query block selection for one request.

    ``idx_q`` is ``[nq, nidx, idx_dim]`` (normed+roped), ``k_all`` the dense
    index keys ``[seq_len, idx_dim]`` of the request, ``q_pos`` the absolute
    position of each query token.  Returns ``[nidx, nq, topk]`` left-packed
    block ids with ``-1`` padding, the contract the attend twin consumes.
    """
    nq, nidx = idx_q.shape[0], idx_q.shape[1]
    seq_len = k_all.shape[0]
    scores = torch.matmul(idx_q.float().transpose(0, 1), k_all.float().t())
    key_pos = torch.arange(seq_len, device=idx_q.device)
    causal = key_pos[None, :] <= q_pos[:, None]                  # [nq, S]
    scores = torch.where(causal[None], scores,
                         torch.full_like(scores, float("-inf")))

    num_blocks = _num_pages(seq_len)
    # Pad the tail block with -inf so its pool sees only real positions.
    pad = num_blocks * _SPARSE_BLOCK_SIZE - seq_len
    if pad:
        scores = torch.cat(
            (scores, torch.full((nidx, nq, pad), float("-inf"),
                                dtype=scores.dtype, device=scores.device)),
            dim=-1)
    block_scores = scores.reshape(nidx, nq, num_blocks,
                                  _SPARSE_BLOCK_SIZE).amax(dim=-1)
    # A token only ever sees blocks up to its own (fully masked blocks score
    # -inf exactly as the kernel leaves them).
    valid_blocks = (q_pos + _SPARSE_BLOCK_SIZE) // _SPARSE_BLOCK_SIZE
    blk = torch.arange(num_blocks, device=idx_q.device)
    visible = blk[None, :] < valid_blocks[:, None]               # [nq, nb]
    block_scores = torch.where(visible[None], block_scores,
                               torch.full_like(block_scores, float("-inf")))
    # Forced blocks: the shipped constexprs give init blocks 1e30 and the
    # trailing local blocks 1e29 (local wins when they coincide, as in the
    # decode kernel).
    if init_blocks > 0:
        forced = visible & (blk < init_blocks)[None, :]
        block_scores = torch.where(forced[None],
                                   torch.full_like(block_scores, 1e30),
                                   block_scores)
    if local_blocks > 0:
        # No clamp: a start below 0 simply makes every visible block local,
        # which is the required behaviour (and a Long clamp is rejected on LPU).
        local_start = valid_blocks - local_blocks
        local = visible & (blk[None, :] >= local_start[:, None])
        block_scores = torch.where(local[None],
                                   torch.full_like(block_scores, 1e29),
                                   block_scores)
    sel = _select_blocks(block_scores, topk)
    if _DEBUG_SELECT:
        _dump_selection(sel, block_scores, seq_len)
    return sel


def _dump_selection(sel: torch.Tensor, block_scores: torch.Tensor,
                    seq_len: int) -> None:
    """Record the last query row's block ids and scores, when asked.

    The block choice is a discrete decision on a max over a whole block, so a
    single near-tie swaps 128 keys at once.  Dumping the scores is what lets an
    LPU-vs-reference logit delta be attributed to a flipped choice instead of
    guessed at from the delta alone.  ``seq_len`` is the keyed length, which
    identifies the prefill (a long prefill arrives in several query chunks).
    """
    global _debug_select_calls
    _debug_select_calls += 1
    last = block_scores[:, -1, :].detach().float().cpu().tolist()
    row = {"call": _debug_select_calls,
           "nq": int(block_scores.shape[1]),
           "seq_len": seq_len,
           "num_blocks": int(block_scores.shape[-1]),
           "block_scores": last,
           "sel": sel[:, -1, :].detach().to(torch.int32).cpu().tolist()}
    with open(_DEBUG_SELECT, "a") as f:
        f.write(json.dumps(row) + "\n")


def _block_sparse_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel: torch.Tensor,
    q_pos: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over the selected blocks for one request.

    ``q`` is ``[nq, num_heads, head_dim]``, ``k``/``v`` are ``[seq_len,
    num_kv_heads, head_dim]``, ``sel`` is ``[nidx, nq, topk]`` from
    ``_index_select_blocks``.  Dense over the request's keys with an additive
    mask (selected block AND causal); ``-1`` selections are ignored, so a block
    can never be double-counted.
    """
    nq, seq_len = q.shape[0], k.shape[0]
    num_blocks = _num_pages(seq_len)
    key_pos = torch.arange(seq_len, device=q.device)
    group = num_heads // num_kv_heads

    keep = _block_keep(sel, num_blocks)                          # [nidx, nq, nb]
    keep = keep[..., key_pos // _SPARSE_BLOCK_SIZE]              # [nidx, nq, S]
    keep = keep & (key_pos[None, :] <= q_pos[:, None])[None]
    keep = keep.repeat_interleave(group, dim=0)                  # [nh, nq, S]

    qh = q.float().transpose(0, 1)                                # [nh, nq, D]
    kh = k.float().transpose(0, 1).repeat_interleave(group, dim=0)
    vh = v.float().transpose(0, 1).repeat_interleave(group, dim=0)
    scores = torch.bmm(qh, kh.transpose(1, 2)) * scale
    scores = torch.where(keep, scores, torch.full_like(scores, float("-inf")))
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs, vh).transpose(0, 1).to(q.dtype)       # [nq, nh, D]


def _select_blocks(block_scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-``topk`` per query row, left-packed, ``-1`` for unreachable blocks.

    ``torch.topk`` is not usable on LPU (its sort is unsound), so this sorts
    descending and slices.  Rows with fewer than ``topk`` visible blocks are
    padded with ``-1`` exactly like the Triton kernel's ``valid_mask``.
    """
    num_blocks = block_scores.shape[-1]
    take = min(topk, num_blocks)
    sorted_scores, order = torch.sort(block_scores, dim=-1, descending=True,
                                      stable=True)
    picked = order[..., :take]
    if take < topk:
        pad = torch.full(picked.shape[:-1] + (topk - take,), -1,
                         dtype=picked.dtype, device=picked.device)
        picked = torch.cat((picked, pad), dim=-1)
    elif take > 0:
        # A block is unreachable when every score in it is -inf (entirely
        # future); the kernel leaves those slots at -1.
        picked = torch.where(sorted_scores[..., :take] > float("-inf"), picked,
                             torch.full_like(picked, -1))
    return picked.to(torch.int32)


# ---------------------------------------------------------------------------
# 3. Lightning indexer (Triton -> pure torch)
# ---------------------------------------------------------------------------
def _make_indexer_impl_cls():
    from vllm.models.minimax_m3.common.indexer import (
        MiniMaxM3IndexerImpl,
        MiniMaxM3IndexerMetadata,
    )
    from vllm.forward_context import get_forward_context

    class _LpuIndexerImpl(MiniMaxM3IndexerImpl):
        """Block-score + block top-k for both prefill and decode.

        The Triton path splits decode into score/partial-topk/merge kernels; the
        twin scans the request's blocks once and selects directly.  Boost rules
        follow the shipped constexprs (``MASK_INIT=MASK_LOCAL=False``): init
        blocks are forced with score ``1e30``, the trailing ``local_blocks``
        blocks with ``1e29`` (they win even over init blocks, matching the
        decode kernel's ``where(is_local, 1e29, where(is_init, 1e30, score))``).
        """

        def _score_and_select(self, idx_q, cache, block_table, seq_lens,
                              context_lens, query_lens, out, out_offset):
            q_rows = 0
            for b in range(seq_lens.shape[0]):
                nq = query_lens[b]
                if nq == 0:
                    continue
                seq_len = int(seq_lens[b])
                ctx = int(context_lens[b])
                pages = block_table[b, :_num_pages(seq_len)].tolist()
                k_all = _gather_index_pages(cache, pages, seq_len)  # [S, D]
                q = idx_q[q_rows:q_rows + nq]                       # [nq, H, D]
                q_pos = torch.arange(ctx, ctx + nq, device=q.device)
                sel = _index_select_blocks(
                    q, k_all, q_pos, self.init_blocks, self.local_blocks,
                    self.topk_blocks)                               # [H, nq, topk]
                out[:, out_offset + q_rows:out_offset + q_rows + nq, :].copy_(
                    sel)
                q_rows += nq

        def forward(self, index_query):
            attn_metadata = get_forward_context().attn_metadata
            if not isinstance(attn_metadata, dict):
                return None, None  # profiling run; caches unbound
            index_md = attn_metadata[self.index_cache.prefix]
            assert isinstance(index_md, MiniMaxM3IndexerMetadata)
            num_tokens = index_md.num_actual_tokens
            nd = index_md.num_decode_tokens
            idx_q = index_query[:num_tokens].view(
                -1, self.num_index_heads, self.index_head_dim)
            cache = self.index_cache.kv_cache

            out = self.topk_indices_buffer
            if out is None:
                out = torch.empty(
                    (self.num_index_heads, num_tokens, self.topk_blocks),
                    dtype=torch.int32, device=idx_q.device)
                self.topk_indices_buffer = out

            decode_topk = prefill_topk = None
            if index_md.num_decodes > 0:
                d = index_md.decode
                assert d is not None
                decode_topk = out[:, :nd, :]
                self._score_and_select(
                    idx_q[:nd], cache, d.block_table, d.seq_lens,
                    d.seq_lens - d.decode_query_len,
                    [d.decode_query_len] * d.seq_lens.shape[0], out, 0)
            if index_md.num_prefills > 0:
                p = index_md.prefill
                assert p is not None
                prefill_topk = out[:, nd:num_tokens, :]
                cu = p.cu_seqlens_q.tolist()
                self._score_and_select(
                    idx_q[nd:], cache, p.block_table, p.seq_lens,
                    p.context_lens, [cu[i + 1] - cu[i] for i in range(
                        len(cu) - 1)], out, nd)
            return decode_topk, prefill_topk

    return _LpuIndexerImpl


# ---------------------------------------------------------------------------
# 4. Block-sparse attention (Triton -> pure torch, dense masked)
# ---------------------------------------------------------------------------
def _make_sparse_impl_cls():
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseImpl,
        MiniMaxM3SparseMetadata,
    )
    from vllm.forward_context import get_forward_context

    class _LpuSparseImpl(MiniMaxM3SparseImpl):
        """Attend the indexer-selected blocks with an additive mask.

        Dense over each request's whole paged KV (O(S^2)) instead of gathering
        just the selected blocks: this is the correctness port.  The selection
        itself is identical to the kernel's -- selected block ids come from the
        shared ``topk_indices_buffer`` and ``-1`` entries are ignored.
        """

        def _attend(self, q, kv_cache, sel, block_table, seq_lens,
                    context_lens, query_lens, out, out_offset):
            q_rows = 0
            for b in range(seq_lens.shape[0]):
                nq = query_lens[b]
                if nq == 0:
                    continue
                seq_len = int(seq_lens[b])
                ctx = int(context_lens[b])
                pages = block_table[b, :_num_pages(seq_len)].tolist()
                k, v = _gather_kv_pages(kv_cache, pages, seq_len)    # [S, nkv, D]
                q_b = q[q_rows:q_rows + nq]                          # [nq, nh, D]
                q_pos = torch.arange(ctx, ctx + nq, device=q.device)
                res = _block_sparse_attend(
                    q_b, k, v, sel[:, out_offset + q_rows:
                                   out_offset + q_rows + nq, :], q_pos,
                    self.num_heads, self.num_kv_heads, self.scale)
                out[out_offset + q_rows:out_offset + q_rows + nq].copy_(res)
                q_rows += nq

        def forward(self, layer, query, kv_cache, output):
            attn_metadata = get_forward_context().attn_metadata
            if not isinstance(attn_metadata, dict):
                return output  # profiling run; caches unbound
            main_md = attn_metadata[layer.layer_name]
            assert isinstance(main_md, MiniMaxM3SparseMetadata)

            nd = main_md.num_decode_tokens
            num_tokens = main_md.num_actual_tokens
            topk = layer.topk_indices_buffer
            assert topk is not None
            head_dim = self.head_size
            q = query[:num_tokens].view(-1, self.num_heads, head_dim)
            out = output[:num_tokens].view(-1, self.num_heads, head_dim)

            if main_md.num_decodes > 0:
                d = main_md.decode
                assert d is not None
                self._attend(q[:nd], kv_cache, topk, d.block_table, d.seq_lens,
                             d.seq_lens - d.decode_query_len,
                             [d.decode_query_len] * d.seq_lens.shape[0],
                             out, 0)
            if main_md.num_prefills > 0:
                p = main_md.prefill
                assert p is not None
                cu = p.cu_seqlens_q.tolist()
                self._attend(q[nd:], kv_cache, topk, p.block_table, p.seq_lens,
                             p.context_lens,
                             [cu[i + 1] - cu[i] for i in range(len(cu) - 1)],
                             out, nd)
            return output

    return _LpuSparseImpl


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def apply_lpu_minimax_m3_patch() -> None:
    """Rebind the four CUDA/Triton entry points to their LPU twins.

    Idempotent.  Inert unless a MiniMax-M3 model is loaded: the rebinds only
    affect that model's own classes, its impl-selection hooks, and the one
    ``_custom_ops`` wrapper no other model calls.
    """
    global _applied
    if _applied:
        return

    from vllm import _custom_ops as ops
    from vllm.models.minimax_m3.common import indexer as indexer_module
    from vllm.models.minimax_m3.common import sparse_attention as sparse_module
    from vllm.models.minimax_m3.nvidia import model as m3_model

    # Impl classes are picked by module-level hooks at layer construction time;
    # nvidia/model.py imports select_main_impl_cls into its own namespace, so
    # that binding needs rebinding too.  Rebinding a module attribute is silent
    # when the name is gone, so check first and fail loudly on upstream drift.
    for module, symbol in ((indexer_module, "select_indexer_impl_cls"),
                           (sparse_module, "select_main_impl_cls"),
                           (m3_model, "select_main_impl_cls")):
        if not hasattr(module, symbol):
            raise AttributeError(
                f"MiniMax-M3 LPU patch: {module.__name__}.{symbol} is missing; "
                f"upstream changed the impl-selection hook")

    m3_model.MiniMAXGemmaRMSNorm.forward = _forward_lpu
    ops.fused_minimax_m3_qknorm_rope_kv_insert = _fused_qknorm_rope_kv_insert
    _install_swiglu_oai_twin()

    indexer_cls = _make_indexer_impl_cls()
    sparse_cls = _make_sparse_impl_cls()

    def _select_indexer(**_kwargs):
        return indexer_cls

    def _select_sparse(**_kwargs):
        return sparse_cls

    indexer_module.select_indexer_impl_cls = _select_indexer
    sparse_module.select_main_impl_cls = _select_sparse
    m3_model.select_main_impl_cls = _select_sparse

    _applied = True
