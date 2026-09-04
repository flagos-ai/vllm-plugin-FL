# Copyright (c) 2026 BAAI. All rights reserved.
#
# Pure-PyTorch replacements for GDN Triton kernels on Kunlunxin XPU.
#
# These are used to bypass the GDN fused post-conv Triton kernel, whose
# compilation hangs on FlagTree 3.6 + Kunlunxin XPU at TritonXPULegalizePass
# / sortOpTreeBwd. The PyTorch fallback keeps the original operator semantics
# (q/k split + optional L2Norm, v split, g = -exp(A_log)*softplus(a+dt_bias),
# beta = sigmoid(b)) while avoiding compilation of the fused Triton kernel.

import torch
import torch.nn.functional as F


def fused_post_conv_prep_torch(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch replacement for fused_post_conv_prep.

    Matches the semantics of the Triton kernel in
    ``vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py``:
    splits conv_output into q/k/v, optionally L2-normalizes q/k, and computes
    the GDN gating (g/beta) from a/b/A_log/dt_bias.
    """
    L = conv_output.shape[0]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    dtype = conv_output.dtype
    device = conv_output.device

    if L == 0:
        q = torch.empty(L, H, K, dtype=dtype, device=device)
        k = torch.empty(L, H, K, dtype=dtype, device=device)
        v = torch.empty(L, HV, V, dtype=dtype, device=device)
        g = torch.empty(L, HV, dtype=torch.float32, device=device)
        beta_out = torch.empty(L, HV, dtype=torch.float32, device=device)
        return q, k, v, g, beta_out

    HK = H * K
    expected_width = 2 * HK + HV * V
    if conv_output.shape[1] != expected_width:
        raise ValueError(
            f"conv_output width {conv_output.shape[1]} != expected {expected_width}"
        )

    q = conv_output[:, :HK].reshape(L, H, K)
    k = conv_output[:, HK:2 * HK].reshape(L, H, K)
    v = conv_output[:, 2 * HK:].reshape(L, HV, V)

    if apply_l2norm:
        q = l2norm_fwd_torch(q).to(dtype)
        k = l2norm_fwd_torch(k).to(dtype)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # gating: g = -exp(A_log) * softplus(a + dt_bias)
    x = a.float() + dt_bias.float().unsqueeze(0)
    sp = F.softplus(x)
    g = -torch.exp(A_log.float()).unsqueeze(0) * sp  # [L, HV] float32

    if output_g_exp:
        g = torch.exp(g)

    beta_out = torch.sigmoid(b.float())  # [L, HV] float32

    return q, k, v, g, beta_out


def l2norm_fwd_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    x_flat = x.reshape(-1, x.shape[-1]).float()
    y = x_flat * torch.rsqrt(torch.sum(x_flat * x_flat, dim=-1, keepdim=True) + eps)
    return y.reshape(x.shape).to(x.dtype)
