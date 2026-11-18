# SPDX-License-Identifier: Apache-2.0
"""Adapt the vLLM output-buffer contracts to FlagGems-vLLM."""

import torch


def ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    from flaggems_vllm import ihc_pre as impl

    return impl(x, weight, scale, base, magnitude, hc_eps, norm_eps)


def ihc_post(x, residual, post):
    from flaggems_vllm import ihc_post as impl

    return impl(x, residual, post)


def sparse_mla(q, kv, indices, scale, *, attn_sink, topk_length, out=None):
    from flaggems_vllm import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=attn_sink, topk_length=topk_length, out=out
    )


def fp32_router_gemm(x, weight):
    from flaggems_vllm import fp32_router_gemm as impl

    return impl(x, weight)


def paged_mqa(q, cache, weights, lengths, table, width):
    from flaggems_vllm import fp8_fp4_paged_mqa_logits_graph

    # Each row already has its own block table and length. Consume the pages
    # directly in eager and capture, with no host copy of lengths or cache.
    return fp8_fp4_paged_mqa_logits_graph(
        (q, None), cache, weights, lengths, table, None, width, clean_logits=False
    )


def int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    from flaggems_vllm import cutlass_scaled_mm

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
    return out


def concat_mla_q(ql_nope, q_pe, out):
    from flaggems_vllm import concat_mla_q as impl

    impl(ql_nope, q_pe, out)


def concat_and_cache_mla(kv_c, k_pe, cache, slot_mapping, cache_dtype, scale):
    from flaggems_vllm import concat_and_cache_mla as impl

    if k_pe.ndim == 3 and k_pe.shape[1] == 1:
        k_pe = k_pe.squeeze(1)
    impl(kv_c, k_pe, cache, slot_mapping, cache_dtype, scale)


def moe_align(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    total,
    expert_map,
    scatter_idx,
):
    from flaggems_vllm import moe_align_block_size_ep

    moe_align_block_size_ep(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        total,
        expert_map,
        scatter_idx,
    )


def moe_sum(values, out, topk_ids, expert_map):
    from flaggems_vllm import moe_sum as impl

    impl(values, out, topk_ids, expert_map)


def expert_quant_int8(x):
    from flaggems_vllm import moe_per_token_quant_int8

    return moe_per_token_quant_int8(x)


def mrope(
    query, key, cache, positions, section, head_size, rotary_dim, interleaved, neox
):
    from flaggems_vllm import mrope as impl

    q, k = query.contiguous(), key.contiguous()
    impl(
        q,
        k,
        cache.contiguous(),
        positions.contiguous(),
        *section,
        head_size,
        rotary_dim,
        is_interleaved=interleaved,
        is_neox_style=neox,
    )
    if not query.is_contiguous():
        query.copy_(q)
    if not key.is_contiguous():
        key.copy_(k)
    return query, key
