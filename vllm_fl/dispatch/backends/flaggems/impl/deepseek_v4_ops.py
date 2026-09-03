"""
FlagGems deepseekV4 operator implementations.
"""

from __future__ import annotations

import torch

def gather_k_cache_flaggems(
        out, 
        k_cache, 
        seq_lens, 
        gather_lens, 
        block_table, 
        block_size, 
        offset
):
    raise NotImplementedError

def fused_indexer_q_rope_flaggems(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale
):
    raise NotImplementedError

def fused_inv_rope_flaggems(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        quant_group_size,
        tma_aligned_scales,
):
    raise NotImplementedError

def gather_bf16_kv_from_pages_flaggems(
        kv_cache,
        block_table,
        cu_seq_lens,
        token_to_seq,
        total_seq_lens,
        dst,
):
    raise NotImplementedError

def bf16_mqa_logits_flaggems(
        q,
        kv,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits
):
    raise NotImplementedError
