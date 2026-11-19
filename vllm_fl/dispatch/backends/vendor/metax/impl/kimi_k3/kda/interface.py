"""K3 prefill adapter preserving BF16 interfaces and FP32 recurrent state."""

import torch
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.chunk import (
    chunk_kda_with_fused_gate_fwd,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.normalization import l2norm
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.preprocess import can_prepare, prepare


def chunk_kda_with_fused_gate(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    g_bias,
    scale=None,
    initial_state=None,
    output_final_state=False,
    lower_bound=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    if lower_bound != -5.0 or q.shape[-1] != 128:
        raise ValueError(
            "The verified K3 adapter requires head128 and gate lower bound -5"
        )
    if cu_seqlens is None or not output_final_state:
        raise ValueError("K3 prefill requires sequence boundaries and final state")
    if use_qk_l2norm_in_kernel and can_prepare(q, k, v):
        q_fp32, k_fp32, v_fp32 = prepare(q, k, v)
    else:
        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q.contiguous()), l2norm(k.contiguous())
        q_fp32, k_fp32, v_fp32 = q.float(), k.float(), v.float()
    if initial_state is None:
        initial_state = torch.zeros(
            (cu_seqlens.numel() - 1, q.shape[2], v.shape[-1], k.shape[-1]),
            device=q.device,
            dtype=torch.float32,
        )
    output, state = chunk_kda_with_fused_gate_fwd(
        q_fp32,
        k_fp32,
        v_fp32,
        raw_g.contiguous(),
        raw_beta,
        A_log,
        g_bias,
        scale if scale is not None else 128**-0.5,
        initial_state.contiguous(),
        True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return output.to(v.dtype), state
