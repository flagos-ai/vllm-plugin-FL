# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.24 index wrappers with explicit MetaX launch parameters.

Kernel bodies are imported unchanged. No tuner/default/compiler state is modified.
"""
import torch
import triton

from vllm.models.minimax_m3.common.ops.index_topk import (
    _decode_index_score_kernel,
    _index_block_score_kernel,
    _topk_index_kernel as _decorated_topk,
    _topk_index_merge_kernel,
    _topk_index_partial_kernel as _decorated_partial,
)
from vllm.utils.math_utils import round_up

from .jit import launch_handle
from vllm_fl.minimax_m3_config import load

SPARSE_BLOCK_SIZE = 128
_topk_index_kernel = launch_handle(_decorated_topk)
_topk_index_partial_kernel = launch_handle(_decorated_partial)


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        **({"num_stages": 1} if load().kernels.safe_stages else {}),
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the
    result is written into ``out[:, :total_q, :]`` instead of a fresh tensor --
    used to keep the top-k output at a stable address for cudagraph capture.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    _topk_index_kernel[grid_topk](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        BLOCK_SIZE_K=64,
        num_warps=2,
        num_stages=1,
    )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = False
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    # TP=1 spec decode scores a wide 4-head x 4-position query tile per K block;
    # reduce stages to ease memory/register pressure. Keep no-spec and TP=4
    # single-head codegen unchanged.
    score_kwargs = pdl_kwargs.copy()
    if num_idx_heads > 1 and max_decode_query_len > 1:
        score_kwargs.update({"num_warps": 4, "num_stages": 2})

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # split-K over seq blocks; chunk count depends only on shape constants so
    # the grid is fixed within a cuda graph.
    TARGET_GRID = 512
    MAX_NUM_KV_CHUNKS = 256
    # Use the configured max decode length to avoid Triton recompiles when
    # switching between qlen=1 and spec-decode verification batches.
    BLOCK_SIZE_Q = triton.next_power_of_2(max_decode_query_len)
    if load().kernels.safe_head_tile:
        BLOCK_SIZE_Q = max(16 // num_idx_heads, BLOCK_SIZE_Q)
        score_kwargs.update({"num_stages": 1})
    score_ctas_per_chunk = seq_lens.shape[0]
    target = max(
        1,
        min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, score_ctas_per_chunk)),
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)
    grid_score = (seq_lens.shape[0], num_kv_chunks)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        num_kv_chunks=num_kv_chunks,
        USE_PDL=use_pdl,
        **score_kwargs,
    )

    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )
    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * num_idx_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    block_size_t = triton.next_power_of_2(topk)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.float32,
        device=idx_q.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_index_partial_kernel[(batch, num_idx_heads, num_topk_chunks)](
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        USE_PDL=use_pdl,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        BLOCK_SIZE_K=64,
        num_warps=2,
        num_stages=1,
        **pdl_kwargs,
    )
    _topk_index_merge_kernel[(batch, num_idx_heads)](
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_topk_chunks=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_kwargs,
    )
    return topk_idx
