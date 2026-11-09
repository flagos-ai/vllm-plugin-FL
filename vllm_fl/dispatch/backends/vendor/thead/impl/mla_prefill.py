# Copyright (c) 2026 BAAI. All rights reserved.
"""Dense MLA prefill via the PPU FA2 wheel."""

import torch


def is_available():
    try:
        import flash_attn  # noqa: F401

        return hasattr(torch.ops.flash_attn, "_flash_attn_varlen_forward")
    except (ImportError, OSError):
        return False


def mla_prefill_thead(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    return_softmax_lse,
):
    import flash_attn  # noqa: F401

    q, k, v = [
        tensor.contiguous() if tensor.stride(-1) != 1 else tensor
        for tensor in (q, k, v)
    ]
    out, lse, _, _ = torch.ops.flash_attn._flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
        leftpad_k=None,
        seqused_k=None,
        zero_tensors=False,
    )
    return (out, lse) if return_softmax_lse else out
