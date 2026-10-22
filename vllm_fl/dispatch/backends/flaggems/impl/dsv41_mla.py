# Copyright (c) 2026 BAAI. All rights reserved.

"""Adapt vLLM's metadata and prefill output ABI to existing FlagGems-vllm MLA.

The provider already implements both attention kernels. Decode is bound
directly by the EMPTY installer; prefill still needs caller-owned ``out``.
"""


def build_tile_scheduler(builder, num_decode_tokens):
    from flaggems_vllm.ops.flash_mla_with_kvcache import get_mla_metadata

    result = dict.fromkeys(("swaonly", "c4a", "c128a", "c1a", "c2a"))
    if num_decode_tokens:
        for layer_type in builder._layer_types:
            result[layer_type] = get_mla_metadata()[0]
    return result


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None
):
    from flaggems_vllm import flash_mla_sparse_fwd as gems_prefill

    result, max_logits, lse = gems_prefill(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
    )
    if out is not None:
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
        ):
            raise ValueError("Sparse MLA output buffer must match the result")
        out.copy_(result)
        result = out
    return result, max_logits, lse
