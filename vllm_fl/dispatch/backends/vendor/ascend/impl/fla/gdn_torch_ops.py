# Copyright (c) 2026 BAAI. All rights reserved.
#
# Pure-PyTorch replacements for GDN Triton kernels on Ascend NPU.
# These implement the same math as the Triton kernels in:
#   vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
#   vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
#   vllm/model_executor/layers/fla/ops/fused_recurrent.py
#   vllm/model_executor/layers/mamba/gdn_linear_attn.py (fused_gdn_gating)
#   vllm/model_executor/layers/fla/ops/l2norm.py

import torch
import torch.nn.functional as F


def chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch recurrent implementation of chunk_gated_delta_rule.

    Handles GQA where num_v_heads (HV) != num_k_heads (H).
    Vectorized across all HV heads per timestep using batched matmul.
    q, k: [B, T, H, K], v: [B, T, HV, V], g/beta: [B, T, HV]
    state: [N, HV, V, K]
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    groups = HV // H  # value heads per key head

    if scale is None:
        scale = K ** -0.5

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = initial_state.clone().float()  # [N, HV, V, K]
    else:
        h = torch.zeros(N, HV, V, K, dtype=torch.float32, device=q.device)

    # Expand k/q heads to match v heads: [B, T, H, K] -> [B, T, HV, K]
    if groups > 1:
        q_exp = q.repeat_interleave(groups, dim=2)  # [B, T, HV, K]
        k_exp = k.repeat_interleave(groups, dim=2)  # [B, T, HV, K]
    else:
        q_exp = q
        k_exp = k

    if cu_seqlens is not None:
        cu_cpu = cu_seqlens.cpu().tolist()
        for i_n in range(N):
            bos, eos = cu_cpu[i_n], cu_cpu[i_n + 1]
            if bos >= eos:
                continue
            hi = h[i_n]  # [HV, V, K]
            for t in range(bos, eos):
                qt = q_exp[0, t].float() * scale  # [HV, K]
                kt = k_exp[0, t].float()           # [HV, K]
                vt = v[0, t].float()               # [HV, V]
                gt = g[0, t].float()               # [HV]
                bt = beta[0, t].float()            # [HV]

                # Gated decay: h *= exp(g)  [HV, V, K] *= [HV, 1, 1]
                hi = hi * torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
                # h @ k: [HV, V, K] x [HV, K] -> [HV, V]
                hk = torch.bmm(hi, kt.unsqueeze(-1)).squeeze(-1)
                # v' = beta * (v - hk)
                vp = (vt - hk) * bt.unsqueeze(-1)
                # h += v' outer k: [HV, V, 1] x [HV, 1, K]
                hi = hi + torch.bmm(vp.unsqueeze(-1), kt.unsqueeze(-2))
                # o = h @ q: [HV, V, K] x [HV, K] -> [HV, V]
                ot = torch.bmm(hi, qt.unsqueeze(-1)).squeeze(-1)
                o[0, t] = ot.to(o.dtype)

            h[i_n] = hi
    else:
        for i_n in range(N):
            hi = h[i_n]
            for t in range(T):
                qt = q_exp[i_n, t].float() * scale
                kt = k_exp[i_n, t].float()
                vt = v[i_n, t].float()
                gt = g[i_n, t].float()
                bt = beta[i_n, t].float()

                hi = hi * torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
                hk = torch.bmm(hi, kt.unsqueeze(-1)).squeeze(-1)
                vp = (vt - hk) * bt.unsqueeze(-1)
                hi = hi + torch.bmm(vp.unsqueeze(-1), kt.unsqueeze(-2))
                ot = torch.bmm(hi, qt.unsqueeze(-1)).squeeze(-1)
                o[i_n, t] = ot.to(o.dtype)

            h[i_n] = hi

    final_state = h if output_final_state else None
    return o, final_state


def l2norm_fwd_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    x_flat = x.reshape(-1, x.shape[-1]).float()
    y = x_flat * torch.rsqrt(torch.sum(x_flat * x_flat, dim=-1, keepdim=True) + eps)
    return y.reshape(x.shape).to(x.dtype)


def fused_gdn_gating_torch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch replacement for the fused_gdn_gating Triton kernel.

    Computes:
        g = -exp(A_log) * softplus(a + dt_bias)
        beta_output = sigmoid(b)
    """
    x = a.float() + dt_bias.float().unsqueeze(0)
    sp = F.softplus(x, beta=beta, threshold=threshold)
    g = -torch.exp(A_log.float()).unsqueeze(0) * sp
    beta_output = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_output.unsqueeze(0)


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
    """Pure-PyTorch replacement for fused_post_conv_prep Triton kernel."""
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

    # Split conv_output into q, k, v components
    q_flat = conv_output[:, :HK]                        # [L, H*K]
    k_flat = conv_output[:, HK:2*HK]                    # [L, H*K]
    v_flat = conv_output[:, 2*HK:2*HK + HV*V]          # [L, HV*V]

    # Reshape to head layout
    q = q_flat.reshape(L, H, K)
    k = k_flat.reshape(L, H, K)
    v = v_flat.reshape(L, HV, V)

    if apply_l2norm:
        q = l2norm_fwd_torch(q).to(dtype)
        k = l2norm_fwd_torch(k).to(dtype)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Gating: g = -exp(A_log) * softplus(a + dt_bias)
    x = a.float() + dt_bias.float().unsqueeze(0)    # [L, HV]
    sp = F.softplus(x)
    g = -torch.exp(A_log.float()).unsqueeze(0) * sp  # [L, HV]

    if output_g_exp:
        g = torch.exp(g)

    beta_out = torch.sigmoid(b.float())  # [L, HV]

    return q, k, v, g, beta_out
