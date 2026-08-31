# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure-torch mHC implementations that inductor can trace and fuse.

Replaces the tilelang-JIT mHC entry points (which required opaque
custom-op wrappers under torch.compile, costing per-graph static-buffer
memory and per-op launch overhead). Math follows upstream:
  - mhc_pre_torch / mhc_post_torch (vllm/model_executor/kernels/mhc/torch.py)
  - norm_weight fusion semantics from the tilelang big_fuse kernel
  - hc_head math from hc_head_reduce_triton_kernel (mhc/triton.py)

Numerical layout notes: fp32 accumulation everywhere, bf16 outputs,
matching the tilelang kernels' documented contracts.
"""

import torch


def _rmsnorm_nw(x: torch.Tensor, eps: float) -> torch.Tensor:
    """No-weight RMSNorm over the last dim, fp32 math."""
    xf = x.to(torch.float32)
    return xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)


def _mhc_mixes(
    residual_flat: torch.Tensor,  # [T, hc_mult, H] bf16
    fn: torch.Tensor,             # [hc_mult3, hc_mult*H] fp32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    num_tokens, hc_mult, hidden_size = residual_flat.shape
    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    pre_mix = pre_mix + hc_pre_eps

    post_mix = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
    comb_logits = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :].view(
        1, hc_mult, hc_mult
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return pre_mix, post_mix, comb_mix


def _layer_input(
    pre_mix: torch.Tensor,          # [T, hc_mult] fp32
    residual_flat: torch.Tensor,    # [T, hc_mult, H]
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    li = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    )
    if norm_weight is not None:
        # Match the tilelang big_fuse rounding order exactly: the rsqrt is
        # computed from the fp32 accumulation, but the value it scales has
        # already been rounded to bf16 (pass-1 stashes bf16, pass-2 rescales).
        rsqrt = torch.rsqrt(li.square().mean(dim=-1, keepdim=True) + norm_eps)
        li = li.to(torch.bfloat16).to(torch.float32) * rsqrt
        li = li * norm_weight.to(torch.float32)
    return li.to(torch.bfloat16)


def mhc_pre_native(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
):
    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    pre_mix, post_mix, comb_mix = _mhc_mixes(
        residual_flat, fn, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    )
    layer_input = _layer_input(pre_mix, residual_flat, norm_weight, norm_eps)
    return (
        post_mix.view(*outer, hc_mult, 1),
        comb_mix.view(*outer, hc_mult, hc_mult),
        layer_input.view(*outer, hidden_size),
    )


def mhc_post_native(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_fused_post_pre_native(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
):
    residual_cur = mhc_post_native(x, residual, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input = mhc_pre_native(
        residual_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        norm_weight=norm_weight, norm_eps=norm_eps,
    )
    return residual_cur, post_mix, comb_mix, layer_input


def hc_head_fused_native(
    hs_flat: torch.Tensor,   # [T, hc_mult, H] bf16
    fn: torch.Tensor,        # [hc_mult, hc_mult*H] fp32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x_flat = hs_flat.flatten(-2)
    x_normed = _rmsnorm_nw(x_flat, rms_eps)
    mixes = torch.nn.functional.linear(x_normed, fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    out = torch.sum(
        pre.unsqueeze(-1) * hs_flat.to(torch.float32), dim=-2
    )
    return out.to(torch.bfloat16)
