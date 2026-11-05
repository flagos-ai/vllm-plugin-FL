# SPDX-License-Identifier: Apache-2.0
"""MiniMax upstream score kernels with torch top-k on Ascend."""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.ops import index_topk as upstream
from vllm.triton_utils import triton

BLOCK = 128


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
    tokens, heads, dim = q.shape
    if heads != num_kv_heads or tokens != seq_lens.shape[0] * decode_query_len:
        raise ValueError("MiniMax index decode shape does not match metadata")
    if decode_query_len > max_decode_query_len:
        raise ValueError("Decode query length exceeds its reserved capacity")
    blocks = triton.cdiv(max_seq_len, BLOCK)
    stride = triton.cdiv(blocks, 16) * 16
    score = torch.full(
        (heads, tokens, stride), float("-inf"), device=q.device, dtype=torch.float32
    )
    if blocks and tokens:
        # Reuse the score-only kernel, avoiding the upstream bitonic merge.
        upstream._decode_index_score_kernel[(seq_lens.shape[0], 1)](
            q,
            cache,
            score,
            block_table,
            seq_lens,
            heads,
            dim,
            init_blocks,
            local_blocks,
            decode_query_len,
            *q.stride(),
            *cache.stride(),
            *score.stride(),
            block_table.stride(0),
            BLOCK_SIZE_K=BLOCK,
            BLOCK_SIZE_Q=triton.next_power_of_2(max_decode_query_len),
            num_kv_chunks=1,
            USE_PDL=False,
        )
    request = torch.arange(tokens, device=q.device) // decode_query_len
    offset = torch.arange(tokens, device=q.device) % decode_query_len
    visible_tokens = (seq_lens[request] - decode_query_len + offset + 1).clamp(min=0)
    valid = ((visible_tokens + BLOCK - 1) // BLOCK).clamp(max=blocks)
    # The decode score kernel already applies init/local forcing.
    return _store(_select(score[..., :blocks], valid, topk), out)


def index_prefill(
    q, cache, metadata, num_kv_heads, topk, init_blocks, local_blocks, out=None
):
    score = upstream.minimax_m3_index_score(
        q,
        cache,
        metadata.block_table,
        metadata.cu_seqlens_q,
        metadata.seq_lens,
        metadata.context_lens,
        metadata.max_query_len,
        metadata.max_seq_len,
        num_kv_heads,
    )
    # Static-size mapping also handles empty requests without dynamic nonzero.
    token = torch.arange(q.shape[0], device=q.device)
    request = (token[:, None] >= metadata.cu_seqlens_q[None, 1:]).sum(-1).long()
    offset = token - metadata.cu_seqlens_q[request]
    valid = (metadata.context_lens[request] + offset + BLOCK) // BLOCK
    blocks = triton.cdiv(metadata.max_seq_len, BLOCK)
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
