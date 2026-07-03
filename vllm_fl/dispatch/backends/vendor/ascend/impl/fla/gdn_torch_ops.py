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

_CHUNK_SIZE = 64


def _chunked_gdr_single_seq(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked GDN delta-rule for a single sequence.

    All inputs are [1, T, H, D] layout (batch=1).
    Processes tokens in chunks of 64 using batched matmul —
    far fewer kernel launches than the per-timestep loop.

    Based on Huawei vllm-ascend's _torch_chunk_gated_delta_rule_chunked.
    """
    chunk_size = _CHUNK_SIZE
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1, eps=1e-6).to(query.dtype)
        key = F.normalize(key, p=2, dim=-1, eps=1e-6).to(key.dtype)

    # Transpose to [B, H, T, D] and cast to float32 for precision
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Cumulative gating within chunks
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # Intra-chunk attention with triangular solve (WY representation)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # Initialize recurrent state
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, v_head_dim, k_head_dim,
            device=value.device, dtype=value.dtype,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # Inter-chunk recurrence — iterates over T/64 chunks (not T timesteps)
    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i = query[:, :, i]   # [B, H, chunk_size, K]
        k_i = key[:, :, i]
        v_i = value[:, :, i]

        attn_inter_chunk = (
            q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        ).masked_fill_(mask_upper, 0)

        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state.transpose(-1, -2)
        v_new = v_i - v_prime
        inter_state = (
            (q_i * g[:, :, i, :, None].exp())
            @ last_recurrent_state.transpose(-1, -2)
        )
        core_attn_out[:, :, i] = inter_state + attn_inter_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + v_new.transpose(-1, -2)
            @ (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
        )

    if not output_final_state:
        last_recurrent_state = None

    # Reshape back and trim padding
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Back to [B, T, H, D]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


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
    """Chunked PyTorch implementation of chunk_gated_delta_rule.

    Processes tokens in chunks of 64 using batched matmul operations,
    reducing NPU kernel launches from O(T) to O(T/64) per layer.

    Handles GQA where num_v_heads (HV) != num_k_heads (H).
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
        states = initial_state.clone().float()  # [N, HV, V, K]
    else:
        states = torch.zeros(N, HV, V, K, dtype=torch.float32, device=q.device)

    # Expand q/k heads to match v heads for GQA
    if groups > 1:
        q_exp = q.repeat_interleave(groups, dim=2)  # [B, T, HV, K]
        k_exp = k.repeat_interleave(groups, dim=2)  # [B, T, HV, K]
    else:
        q_exp = q
        k_exp = k

    # Process each sequence using the chunked algorithm
    if cu_seqlens is not None:
        cu_cpu = cu_seqlens.cpu().tolist()
        for i_n in range(N):
            bos, eos = cu_cpu[i_n], cu_cpu[i_n + 1]
            seq_len = eos - bos
            if seq_len <= 0:
                continue
            b_idx = 0 if (cu_seqlens is not None and B == 1) else i_n

            q_seq = q_exp[b_idx, bos:eos].unsqueeze(0)   # [1, T_seq, HV, K]
            k_seq = k_exp[b_idx, bos:eos].unsqueeze(0)   # [1, T_seq, HV, K]
            v_seq = v[b_idx, bos:eos].unsqueeze(0)        # [1, T_seq, HV, V]
            g_seq = g[b_idx, bos:eos].unsqueeze(0)        # [1, T_seq, HV]
            beta_seq = beta[b_idx, bos:eos].unsqueeze(0)  # [1, T_seq, HV]
            init_seq = states[i_n].unsqueeze(0)           # [1, HV, V, K]

            out_seq, final_state = _chunked_gdr_single_seq(
                query=q_seq, key=k_seq, value=v_seq,
                g=g_seq, beta=beta_seq,
                initial_state=init_seq,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            o[b_idx, bos:eos] = out_seq[0]
            if final_state is not None:
                states[i_n] = final_state[0]
    else:
        for i_n in range(B):
            q_seq = q_exp[i_n:i_n+1]   # [1, T, HV, K]
            k_seq = k_exp[i_n:i_n+1]
            v_seq = v[i_n:i_n+1]
            g_seq = g[i_n:i_n+1]
            beta_seq = beta[i_n:i_n+1]
            init_seq = states[i_n].unsqueeze(0)

            out_seq, final_state = _chunked_gdr_single_seq(
                query=q_seq, key=k_seq, value=v_seq,
                g=g_seq, beta=beta_seq,
                initial_state=init_seq,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            o[i_n] = out_seq[0]
            if final_state is not None:
                states[i_n] = final_state[0]

    if output_final_state:
        return o, states
    return o, None


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softplus."""
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))


def l2norm_fwd_torch(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)


def fused_gdn_gating_torch(
    gate: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Fused sigmoid gating: output = sigmoid(gate) * x"""
    return torch.sigmoid(gate.float()).to(x.dtype) * x


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
    """Pure-PyTorch fused post-conv preparation for GDN prefill.

    Splits conv output into q, k, v, computes gating g and beta.
    conv_output: [L, H*K + H*K + HV*V]
    a: [L, HV] (pre-sigmoid gate input)
    b: [L, HV] (pre-sigmoid beta input)
    A_log: [HV] (log of decay rate)
    dt_bias: [HV] (bias for gate computation)
    num_k_heads: number of K heads (H)
    head_k_dim: dimension per K head (K)
    head_v_dim: dimension per V head (V)
    """
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]  # num value heads derived from A_log shape

    dtype = conv_output.dtype
    device = conv_output.device
    L = conv_output.shape[0]

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
    sp = _softplus(x)
    g = -torch.exp(A_log.float()).unsqueeze(0) * sp  # [L, HV]

    if output_g_exp:
        g = torch.exp(g)

    beta_out = torch.sigmoid(b.float())  # [L, HV]

    return q, k, v, g, beta_out
