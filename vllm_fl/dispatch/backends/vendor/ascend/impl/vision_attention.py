# Copyright (c) 2026 BAAI. All rights reserved.

"""Fused SDPA for the validated BF16 SAM self-attention shape."""

import torch


def supports_scaled_dot_product_attention(query, key, value, attn_bias=None):
    return (
        query.device.type in ("npu", "privateuseone")
        and query.dtype == torch.bfloat16
        and query.ndim == 4
        and query.shape[0] > 0
        and query.shape[1] == 12
        and query.shape[2] > 0
        and query.shape[3] == 64
        and key.shape == query.shape == value.shape
        and key.device == query.device == value.device
        and key.dtype == query.dtype == value.dtype
        and (
            attn_bias is None
            or (
                attn_bias.device == query.device
                and attn_bias.dtype == query.dtype
                and attn_bias.shape == (*query.shape[:3], key.shape[2])
            )
        )
    )


def scaled_dot_product_attention_ascend(query, key, value, attn_bias=None):
    import torch_npu

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        pse_shift=None if attn_bias is None else attn_bias.contiguous(),
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        input_layout="BNSD",
        scale=query.shape[-1] ** -0.5,
    )
    return output
