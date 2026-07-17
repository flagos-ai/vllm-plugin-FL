# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA implementations for mHC (Multi-Head Convolution) operators.
"""

import torch


def mhc_pre_cuda(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA implementation of mhc_pre using vLLM's tilelang kernel.

    Returns:
        post_mix: shape (..., hc_mult, 1), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    return torch.ops.vllm.mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=n_splits,
    )


def mhc_post_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA implementation of mhc_post using vLLM's tilelang kernel.

    Returns:
        out: shape same as residual, dtype torch.bfloat16
    """
    return torch.ops.vllm.mhc_post(x, residual, post, comb)


def hc_head_fused_kernel_cuda(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """
    CUDA implementation of hc_head_fused_kernel using vLLM's tilelang kernel.
    Mutates `out` in-place.
    """
    torch.ops.vllm.hc_head_fused_kernel(
        hs_flat, fn, hc_scale, hc_base, out,
        hidden_size, rms_eps, hc_eps, hc_mult,
    )
