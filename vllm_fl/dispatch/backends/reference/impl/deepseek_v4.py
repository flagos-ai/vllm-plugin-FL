# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference implementations of DeepSeek-V4-specific operators."""

from __future__ import annotations

import torch


def deepseek_v4_inv_rope_quant_int8_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply inverse RoPE and symmetric per-group-row INT8 quantization."""
    if o.ndim != 3:
        raise ValueError("o must be a 3D [tokens, heads, head_dim] tensor")
    tokens, heads, head_dim = o.shape
    if heads != n_groups * heads_per_group:
        raise ValueError("heads must equal n_groups * heads_per_group")
    if head_dim != nope_dim + rope_dim:
        raise ValueError("head_dim must equal nope_dim + rope_dim")
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")

    selected_cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    half_rope = rope_dim // 2
    cos = selected_cache[:, :half_rope].unsqueeze(1).to(torch.float32)
    sin = selected_cache[:, half_rope:rope_dim].unsqueeze(1).to(torch.float32)

    values = o.to(torch.float32)
    nope = values[..., :nope_dim]
    rope = values[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    inv_rope = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin),
        dim=-1,
    ).flatten(-2)
    restored = torch.cat((nope, inv_rope), dim=-1).to(torch.bfloat16)

    group_dim = heads_per_group * head_dim
    grouped = (
        restored.reshape(tokens, n_groups, group_dim)
        .permute(1, 0, 2)
        .contiguous()
        .to(torch.float32)
    )
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scales = absmax / 127.0
    normalized = grouped / scales
    rounded = torch.where(
        normalized >= 0,
        torch.floor(normalized + 0.5),
        torch.ceil(normalized - 0.5),
    )
    quantized = rounded.clamp(-127, 127).to(torch.int8)
    return quantized, scales


def deepseek_v4_int8_scaled_mm_torch(x, weight, scale_a, scale_b, out_dtype, bias=None):
    out = (x.float() @ weight.float()) * scale_a.float()
    out = out * scale_b.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _rms_norm(x, weight, eps):
    out = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


def deepseek_v4_mhc_pre_torch(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    post, comb, layer_input = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    if norm_weight is not None:
        layer_input = _rms_norm(layer_input, norm_weight, norm_eps)
    return post, comb, layer_input


def deepseek_v4_mhc_post_torch(x, residual, post_mix, comb_mix):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    return mhc_post_torch(x, residual, post_mix, comb_mix)


def deepseek_v4_mhc_fused_post_pre_torch(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    del tile_n
    residual_cur = deepseek_v4_mhc_post_torch(x, residual, post_mix, comb_mix)
    post_cur, comb_cur, layer_input = deepseek_v4_mhc_pre_torch(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        norm_weight,
        norm_eps,
    )
    return residual_cur, post_cur, comb_cur, layer_input


def deepseek_v4_hc_head_torch(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    x = hs_flat.flatten(-2).float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(torch.nn.functional.linear(x, fn) * hc_scale + hc_base)
    pre = pre + hc_eps
    return torch.sum(pre.unsqueeze(-1) * hs_flat.float(), dim=-2).to(torch.bfloat16)


def deepseek_v4_inv_rope_quant_fp8_torch(
    o,
    positions,
    cos_sin_cache,
    heads_per_group,
    quant_group_size,
    chunks_per_head,
    rope_start,
    half_rope,
    tma_aligned_scales,
    fp8_max,
    tma_aligned_T,
    num_tokens,
    n_groups,
    d,
    scale_inner,
):
    del chunks_per_head
    if tma_aligned_scales:
        raise NotImplementedError(
            "reference FP8 inverse-RoPE does not pack UE8M0 scales"
        )
    cache = cos_sin_cache.index_select(0, positions.long())
    values = o.float()
    rope = values[..., rope_start : rope_start + 2 * half_rope]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    cos = cache[:, :half_rope].unsqueeze(1).float()
    sin = cache[:, half_rope : 2 * half_rope].unsqueeze(1).float()
    restored_rope = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1
    ).flatten(-2)
    restored = torch.cat(
        (
            values[..., :rope_start],
            restored_rope,
            values[..., rope_start + 2 * half_rope :],
        ),
        dim=-1,
    ).to(torch.bfloat16)
    grouped = restored.reshape(num_tokens, n_groups, d).permute(1, 0, 2)
    padded = torch.nn.functional.pad(
        grouped.float(),
        (0, scale_inner * quant_group_size - d),
    ).reshape(n_groups, num_tokens, scale_inner, quant_group_size)
    scales = padded.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
    expanded = scales.repeat_interleave(quant_group_size, dim=-1)[..., :d]
    fp8_buf = (
        (grouped.float() / expanded).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    )
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=torch.float32,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    scale_buf.copy_(scales)
    return fp8_buf, scale_buf
