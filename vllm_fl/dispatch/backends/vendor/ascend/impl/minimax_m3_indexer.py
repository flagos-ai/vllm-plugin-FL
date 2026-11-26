# SPDX-License-Identifier: Apache-2.0
"""MiniMax upstream score kernels with torch top-k on Ascend."""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_score,
)

BLOCK = SPARSE_BLOCK_SIZE


def _select(score, valid, topk, init_blocks=0, local_blocks=0):
    heads, tokens, blocks = score.shape
    if blocks == 0:
        return torch.full(
            (heads, tokens, topk), -1, dtype=torch.int32, device=score.device
        )
    positions = torch.arange(blocks, device=score.device)
    visible = positions[None, :] < valid[:, None]
    if init_blocks or local_blocks:
        initial = positions[None, :] < init_blocks
        local = positions[None, :] >= (valid - local_blocks).clamp(min=0)[:, None]
        forced = torch.where(local, 1e29, 1e30)
        score = torch.where((initial | local)[None, :, :], forced[None, :, :], score)
    score = score.masked_fill(~visible[None, :, :], float("-inf"))
    count = min(topk, blocks)
    indices = torch.topk(score, count, dim=-1, sorted=True).indices
    indices = indices.masked_fill(indices >= valid[None, :, None], -1).to(torch.int32)
    if count < topk:
        indices = torch.cat(
            (
                indices,
                torch.full(
                    (heads, tokens, topk - count),
                    -1,
                    dtype=torch.int32,
                    device=score.device,
                ),
            ),
            -1,
        )
    return indices


def _store(result, out):
    if out is None:
        return result
    view = out[:, : result.shape[1], :]
    view.copy_(result)
    return view


def index_decode(
    q,
    cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out=None,
):
    tokens, heads, _ = q.shape
    if heads != num_kv_heads or tokens != seq_lens.shape[0] * decode_query_len:
        raise ValueError("MiniMax index decode shape does not match metadata")
    if decode_query_len > max_decode_query_len:
        raise ValueError("Decode query length exceeds its reserved capacity")
    query_starts = (
        torch.arange(seq_lens.shape[0] + 1, device=q.device, dtype=torch.int32)
        * decode_query_len
    )
    return _index(
        q,
        cache,
        block_table,
        query_starts,
        seq_lens,
        (seq_lens - decode_query_len).clamp(min=0),
        max_decode_query_len,
        max_seq_len,
        num_kv_heads,
        topk,
        init_blocks,
        local_blocks,
        out,
    )


def index_prefill(
    q, cache, metadata, num_kv_heads, topk, init_blocks, local_blocks, out=None
):
    return _index(
        q,
        cache,
        metadata.block_table,
        metadata.cu_seqlens_q,
        metadata.seq_lens,
        metadata.context_lens,
        metadata.max_query_len,
        metadata.max_seq_len,
        num_kv_heads,
        topk,
        init_blocks,
        local_blocks,
        out,
    )


def _index(
    q,
    cache,
    block_table,
    query_starts,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
    topk,
    init_blocks,
    local_blocks,
    out,
):
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    if not blocks or not q.shape[0]:
        result = q.new_full((num_kv_heads, q.shape[0], topk), -1, dtype=torch.int32)
        return _store(result, out)
    score = minimax_m3_index_score(
        q,
        cache,
        block_table,
        query_starts,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads,
    )
    # Static-size mapping also handles empty requests without dynamic nonzero.
    token = torch.arange(q.shape[0], device=q.device)
    request = (token[:, None] >= query_starts[None, 1:]).sum(-1).long()
    offset = token - query_starts[request]
    length = torch.minimum(prefix_lens[request] + offset + 1, seq_lens[request])
    valid = (length.clamp_min(0) + BLOCK - 1) // BLOCK
    valid = valid.clamp(max=blocks)
    return _store(
        _select(score[..., :blocks], valid, topk, init_blocks, local_blocks), out
    )


def index_forward(impl, query):
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None, None
    md = metadata[impl.index_cache.prefix]
    nd = md.num_decode_tokens
    query = query[: md.num_actual_tokens].view(
        -1, impl.num_index_heads, impl.index_head_dim
    )
    cache, out = impl.index_cache.kv_cache, impl.topk_indices_buffer
    decode = prefill = None
    if md.num_decodes:
        d = md.decode
        decode = index_decode(
            query[:nd],
            cache,
            d.block_table,
            d.seq_lens,
            d.max_seq_len,
            impl.topk_blocks,
            impl.init_blocks,
            impl.local_blocks,
            impl.num_kv_heads,
            d.decode_query_len,
            d.max_decode_query_len,
            out=out,
        )
    if md.num_prefills:
        prefill = index_prefill(
            query[nd:],
            cache,
            md.prefill,
            impl.num_kv_heads,
            impl.topk_blocks,
            impl.init_blocks,
            impl.local_blocks,
            out=out[:, nd:, :] if out is not None else None,
        )
    return decode, prefill
