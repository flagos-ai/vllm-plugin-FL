# SPDX-License-Identifier: Apache-2.0
"""Torch reference contracts for continuous paged DeepSeek V4 caches.

Projection, page allocation, query metadata and output-row scheduling belong to
the caller. These providers own numerical work and cache writes only; they do
not depend on an Ascend extension, runner or process group.
"""

import torch
import torch.nn.functional as F


def deepseek_v4_partial_rope(x, positions, cos_sin_cache, inverse=False):
    """Rotate the final head dimensions in FP32 with adjacent-pair convention."""
    width = cos_sin_cache.shape[-1]
    if x.ndim not in (2, 3) or positions.shape != x.shape[:1]:
        raise ValueError("Expected [tokens, (heads), dim] and one position per token")
    if width <= 0 or width % 2 or width > x.shape[-1]:
        raise ValueError("RoPE width must be positive, even and within the head")
    cache = cos_sin_cache[positions.long().clamp(0, cos_sin_cache.shape[0] - 1)]
    cos, sin = cache.float().chunk(2, -1)
    if x.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    if inverse:
        sin = -sin
    even, odd = x[..., -width::2].float(), x[..., -width + 1 :: 2].float()
    rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    result = torch.cat([x[..., :-width], rotated.flatten(-2).to(x.dtype)], -1)
    valid = (positions >= 0) & (positions < cos_sin_cache.shape[0])
    return torch.where(valid.reshape(-1, *([1] * (x.ndim - 1))), result, 0)


def deepseek_v4_hadamard(x, matrix):
    """Keep the projection and normalized scaling as separate dtype-round stages."""
    dim = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < dim:
        raise ValueError("Expected a covering square Hadamard matrix")
    padded = F.pad(x, (0, matrix.shape[0] - dim))
    return (F.linear(padded, matrix.to(x.dtype)) * dim**-0.5)[..., :dim]


def deepseek_v4_index_quant(x):
    from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8

    if x.ndim < 2 or not x.is_floating_point():
        raise ValueError("Indexer quantization requires floating token/head rows")
    quantized, scale = dynamic_per_token_quant_int8(x.reshape(-1, x.shape[-1]))
    return quantized.reshape(x.shape), scale.reshape(*x.shape[:-1], 1).half()


def deepseek_v4_grouped_project(x, weight, groups):
    """Apply logical checkpoint [groups*out, in] weights, not a dense shared GEMM."""
    if x.ndim != 3 or groups < 1 or weight.ndim != 2:
        raise ValueError(
            "Grouped projection requires [tokens, heads, dim] and [out,in]"
        )
    if weight.shape[0] % groups or x.shape[1] * x.shape[2] != groups * weight.shape[1]:
        raise ValueError("Grouped output projection dimensions do not match")
    grouped = x.reshape(x.shape[0], groups, weight.shape[1])
    weights = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1])
    return torch.einsum("tgi,goi->tgo", grouped, weights).flatten(1)


def deepseek_v4_hc_pre(x, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters):
    """Return native-order layer input, post coefficients and residual mix matrix."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    if x.ndim != 3 or x.dtype != torch.bfloat16:
        raise ValueError("HC pre requires BF16 [tokens, hc, hidden] residuals")
    hc, dim = x.shape[1:]
    mix = (2 + hc) * hc
    if fn.shape != (mix, hc * dim) or scale.shape != (3,) or base.shape != (mix,):
        raise ValueError("HC pre mixing parameter shapes do not match the residual")
    if any(t.dtype != torch.float32 for t in (fn, scale, base)):
        raise ValueError("HC mixing parameters must remain FP32")
    if sinkhorn_iters < 1:
        raise ValueError("HC requires at least one Sinkhorn iteration")
    if x.shape[0] == 0:
        return (
            x.new_empty((0, dim)),
            x.new_empty((0, hc), dtype=torch.float32),
            x.new_empty((0, hc, hc), dtype=torch.float32),
        )
    post, comb, layer_input = mhc_pre_torch(
        x.contiguous(), fn, scale, base, rms_eps, hc_eps, hc_eps, 2.0, sinkhorn_iters
    )
    return layer_input, post.squeeze(-1), comb


def deepseek_v4_hc_post(x, residual, post, comb):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    if residual.ndim != 3 or x.shape != (residual.shape[0], residual.shape[2]):
        raise ValueError("HC post input and residual shapes do not match")
    if post.shape != residual.shape[:2] or comb.shape != (
        *residual.shape[:2],
        residual.shape[1],
    ):
        raise ValueError("HC post mixing shapes do not match the residual")
    if x.shape[0] == 0:
        return torch.empty_like(residual)
    return mhc_post_torch(x, residual, post.unsqueeze(-1), comb)


def deepseek_v4_hc_head(x, fn, scale, base, rms_eps, hc_eps):
    """Collapse residual streams with normalized, input-dependent head weights."""
    if x.ndim != 3:
        raise ValueError("HC head requires [tokens, hc, hidden] residuals")
    hc, dim = x.shape[1:]
    if fn.shape != (hc, hc * dim) or scale.numel() != 1 or base.shape != (hc,):
        raise ValueError("HC head parameter shapes do not match the residual")
    if any(t.dtype != torch.float32 for t in (fn, scale, base)):
        raise ValueError("HC mixing parameters must remain FP32")
    if x.shape[0] == 0:
        return x.new_empty((0, dim))
    flat = x.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
