"""TP query-row sharding for the Hygon Indexer prefill path.

This module is reached only when the explicit Hygon feature switch is enabled.
The default Indexer path remains unchanged.
"""
import os
import torch
from vllm.triton_utils import triton


def run_row_sharded(chunk, q_bf16, k_bf16, weights, topk_indices_buffer,
                    topk_tokens, *, tp_group):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import (
        bf16_indexer, prefill_topk
    )
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_mqa import prefill_mqa
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import _get_lightop_attention

    m = chunk.token_end - chunk.token_start
    if m == 0:
        return
    world = tp_group.world_size
    rank = tp_group.rank_in_group
    per_rank = (m + world - 1) // world
    owner_lo, owner_hi = rank * per_rank, min((rank + 1) * per_rank, m)
    local = torch.full((per_rank, topk_tokens), -1,
                       dtype=topk_indices_buffer.dtype,
                       device=topk_indices_buffer.device)

    spans = getattr(chunk, 'hygon_request_spans', ())
    split = (1 < len(spans) <= 8 and m >= 1024 and
             sum((qe-qb)*(ke-kb) for qb,qe,kb,ke in spans)
             <= m*k_bf16.shape[0]*0.75)
    if not split:
        spans = ((0, m, 0, k_bf16.shape[0]),)
    starts = chunk.hygon_local_starts if split else chunk.cu_seqlen_ks
    ends = chunk.hygon_local_ends if split else chunk.cu_seqlen_ke
    attention = _get_lightop_attention()
    if attention is None:
        raise RuntimeError('row-sharded Indexer requires LightOp attention')
    logits_buffer = bf16_indexer._get_logits_buffer(q_bf16.device)

    for qb, qe, kb, ke in spans:
        a, b = max(qb, owner_lo), min(qe, owner_hi)
        if a >= b:
            continue
        # A split crossing a rank boundary may be short. Keep the original
        # large-M implementation contract by extending inside this request,
        # then copy only owned rows to the local fixed-size buffer.
        aa, bb = a, b
        if b-a < 128 and qe-qb >= 128:
            aa = min(a, qe-128)
            bb = aa+128
        q = q_bf16[chunk.token_start+aa:chunk.token_start+bb]
        w = weights[chunk.token_start+aa:chunk.token_start+bb].to(torch.float32).contiguous()
        k = k_bf16[kb:ke]
        ss, ee = starts[aa:bb], ends[aa:bb]
        n = ke-kb
        use_triton = bb-aa >= 128 and n <= 8192
        rows = bb-aa if use_triton or bb-aa < 128 else triton.cdiv(bb-aa,128)*128
        cols = n if use_triton or bb-aa < 128 else triton.cdiv(n,128)*128
        storage = logits_buffer[:rows*cols].view(rows,cols)
        if use_triton:
            prefill_mqa(q,k,w,ss,ee,storage)
        else:
            attention.mqa_logits(q,k,w,ss,ee,None,True,storage)
        tmp = torch.empty((bb-aa, topk_tokens), dtype=local.dtype, device=local.device)
        prefill_topk.prefill_topk(
            storage[:bb-aa,:n], ss, ee, tmp,
            already_masked=use_triton or bb-aa >= 128,
            optimize_wide=True,
        )
        local[a-owner_lo:b-owner_lo].copy_(tmp[a-aa:b-aa])

    # All ranks participate even when their local interval is empty. The
    # normal TP communicator supplies graph-compatible RCCL ordering.
    gathered = tp_group.all_gather(local, dim=0)
    topk_indices_buffer[chunk.token_start:chunk.token_end].copy_(gathered[:m])
    if os.getenv('VLLM_FL_HYGON_INDEXER_ROW_SHARD_LOG', '0') == '1':
        print(f'ROW_SHARD rank={rank}/{world} m={m} owned={owner_lo}:{owner_hi} '
              f'local={local.shape} gathered={gathered.shape}', flush=True)
