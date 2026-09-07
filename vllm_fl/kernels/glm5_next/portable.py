# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral GLM5-Next correctness operators.

These implementations intentionally use only PyTorch tensor operations.  They
are the last-resort path for out-of-tree accelerators when neither the NVIDIA
reference kernels nor a matching FlagGems kernel is available.  They preserve
the model semantics, but are not intended to be the final performance path for
long-context serving.
"""

from __future__ import annotations

import math

import torch

INDEX_HEAD_DIM = 128


def get_fp8_dtype_and_max() -> tuple[torch.dtype, float]:
    """Return the portable E4M3 storage dtype and its finite maximum.

    ROCm-family runtimes use the FNUZ encoding (maximum 224); CUDA-like and
    most other runtimes use E4M3FN (maximum 448).  FlagGems remains the
    preferred provider and performs its own platform query.  This helper only
    defines the last-resort PyTorch cache format.
    """
    if (
        getattr(torch.version, "hip", None) is not None
        and hasattr(torch, "float8_e4m3fnuz")
    ):
        return torch.float8_e4m3fnuz, 224.0
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn, 448.0
    raise RuntimeError(
        "No portable E4M3 FP8 dtype is available; install the matching "
        "FlagGems indexer cache operators for this platform."
    )


def safe_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Reference ``lower_bound * sigmoid(exp(A_log) * (g + bias))``."""
    orig_shape = g.shape[:-1]
    g_2d = g.reshape(-1, g.shape[-1])
    heads = A_log.numel()
    if heads * head_k_dim != g_2d.shape[-1]:
        raise ValueError(
            "KDA gate hidden dimension does not match heads * head_k_dim: "
            f"{g_2d.shape[-1]} != {heads} * {head_k_dim}"
        )
    gate = g_2d.float().view(-1, heads, head_k_dim)
    if g_bias is not None:
        gate = gate + g_bias.float().reshape(heads, head_k_dim)
    gate = lower_bound * torch.sigmoid(
        torch.exp(A_log.float()).reshape(1, heads, 1) * gate
    )
    return gate.reshape(*orig_shape, heads, head_k_dim)


def safe_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = torch.float,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Chunk-local inclusive gate cumsum in the log2 units used by FLA."""
    batch, tokens, heads, dim = raw_g.shape
    gate = safe_kda_gate(
        raw_g.reshape(batch, tokens, heads * dim),
        A_log,
        dim,
        g_bias=g_bias,
        lower_bound=lower_bound,
    ).float()
    out = torch.empty_like(gate, dtype=output_dtype or raw_g.dtype)
    rcp_ln2 = 1.0 / math.log(2.0)

    if cu_seqlens is None:
        ranges = [(b, 0, tokens) for b in range(batch)]
    else:
        if batch != 1:
            raise ValueError("Variable-length KDA expects a flattened batch of 1")
        boundaries = cu_seqlens.detach().to("cpu", torch.int64).tolist()
        ranges = [(0, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    for batch_id, seq_start, seq_end in ranges:
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            out[batch_id, chunk_start:chunk_end] = (
                gate[batch_id, chunk_start:chunk_end].cumsum(dim=0) * rcp_ln2
            ).to(out.dtype)
    return out


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


def _recurrent_kda_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one KDA sequence using the reference FP32 recurrence."""
    dtype = v.dtype
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    gate_f, beta_f = gate.float(), beta.float()
    state_f = state.float()
    output = torch.empty_like(v_f)
    for token in range(q.shape[0]):
        q_i = q_f[token] * scale
        k_i = k_f[token]
        v_i = v_f[token]
        gate_i = gate_f[token]
        beta_i = beta_f[token]
        state_f = state_f * torch.exp(gate_i)[..., None]
        residual = v_i - (k_i[..., None] * state_f).sum(dim=-2)
        state_f = state_f + torch.einsum(
            "hk,hv->hkv", beta_i[..., None] * k_i, residual
        )
        output[token] = torch.einsum("hk,hkv->hv", q_i, state_f)
    return output.to(dtype), state_f


def _sequence_ranges(
    batch: int,
    tokens: int,
    cu_seqlens: torch.Tensor | None,
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(batch_id, 0, tokens) for batch_id in range(batch)]
    if batch != 1:
        raise ValueError("Variable-length KDA expects a flattened batch of 1")
    boundaries = cu_seqlens.detach().to("cpu", torch.int64).tolist()
    return [
        (0, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
    ]


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backend-neutral KDA recurrence compatible with vLLM's tensor layout."""
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if scale is None:
        scale = key_dim**-0.5
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    ranges = _sequence_ranges(batch, tokens, cu_seqlens)
    if initial_state is None:
        state_count = len(ranges)
        states = torch.zeros(
            state_count,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        states = initial_state if inplace_final_state else initial_state.clone()

    if ssm_state_indices is None:
        state_indices = list(range(len(ranges)))
    else:
        indices_cpu = ssm_state_indices.detach().to("cpu", torch.int64)
        if indices_cpu.ndim == 1:
            state_indices = indices_cpu.tolist()
        else:
            # vLLM's regular decode path uses one state slot per request.  For
            # speculative metadata use the first accepted slot for the request;
            # the portable path is correctness-oriented and processes tokens in
            # request order.
            state_indices = indices_cpu[:, 0].tolist()

    output = torch.empty_like(v)
    final_states: list[torch.Tensor] = []
    for seq_id, (batch_id, start, end) in enumerate(ranges):
        state_id = int(state_indices[seq_id])
        seq_out, seq_state = _recurrent_kda_sequence(
            q[batch_id, start:end],
            k[batch_id, start:end],
            v[batch_id, start:end],
            gate[batch_id, start:end],
            beta[batch_id, start:end],
            states[state_id],
            float(scale),
        )
        output[batch_id, start:end] = seq_out
        states[state_id].copy_(seq_state)
        final_states.append(seq_state)

    if not output_final_state:
        return output, None
    if initial_state is not None and inplace_final_state:
        return output, initial_state
    if cu_seqlens is not None or batch == len(ranges):
        return output, torch.stack(final_states, dim=0)
    return output, states


def chunk_kda_with_safe_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float = -5.0,
):
    gate = safe_kda_gate(
        raw_g.reshape(*raw_g.shape[:-2], -1),
        A_log,
        raw_g.shape[-1],
        g_bias=g_bias,
        lower_bound=lower_bound,
    )
    return recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del kwargs
    if beta is None:
        raise ValueError("KDA beta tensor is required")
    return recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        inplace_final_state=inplace_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )


def _apply_causal_conv_activation(
    value: torch.Tensor, activation: bool | str | None
) -> torch.Tensor:
    if activation is True or activation in ("silu", "swish"):
        return torch.nn.functional.silu(value)
    if activation in (False, None):
        return value
    raise ValueError(f"Unsupported causal-conv activation: {activation}")


def _causal_conv_sequence(
    sequence: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference depthwise causal convolution for one ``[dim, tokens]`` row."""
    width = weight.shape[-1]
    history = state[..., -(width - 1) :].clone()
    outputs = []
    for token in range(sequence.shape[-1]):
        window = torch.cat((history, sequence[:, token : token + 1]), dim=-1)
        value = (window.float() * weight.float()).sum(dim=-1)
        if bias is not None:
            value = value + bias.float()
        outputs.append(_apply_causal_conv_activation(value, activation))
        history = window[:, 1:]
    if outputs:
        output = torch.stack(outputs, dim=-1).to(sequence.dtype)
    else:
        output = sequence.new_empty(sequence.shape)
    next_state = state.clone()
    next_state[..., -(width - 1) :] = history.to(next_state.dtype)
    return output, next_state


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: bool | str | None = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = -1,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Portable regular-prefill subset of vLLM's causal-conv contract.

    APC cache-block rotation is deliberately rejected rather than silently
    producing a different state.  Normal continuous-batching prefill, which is
    the GLM5-Next serving path, is fully covered.
    """
    del block_size_to_align, metadata, validate_data
    if any(
        value is not None
        for value in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError(
            "Portable GLM5-Next causal_conv1d_fn does not support APC metadata"
        )
    if x.ndim != 2:
        raise ValueError("Portable causal_conv1d_fn expects [dim, total_tokens]")
    boundaries = query_start_loc.detach().to("cpu", torch.int64).tolist()
    output = torch.empty_like(x)
    for request, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state_id = request if cache_indices is None else int(cache_indices[request].item())
        if state_id in (pad_slot_id, null_block_id) or state_id < 0:
            output[:, start:end].zero_()
            continue
        use_state = (
            has_initial_state is None
            or bool(has_initial_state[request].item())
        )
        state = (
            conv_states[state_id]
            if use_state
            else torch.zeros_like(conv_states[state_id])
        )
        request_output, next_state = _causal_conv_sequence(
            x[:, start:end].to(conv_states.dtype),
            state,
            weight,
            bias,
            activation,
        )
        output[:, start:end] = request_output.to(output.dtype)
        conv_states[state_id].copy_(next_state)
    return output


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Portable plain/spec-free decode subset of vLLM's causal-conv update."""
    del max_query_len, validate_data
    if any(
        value is not None
        for value in (
            num_accepted_tokens,
            block_idx_last_scheduled_token,
            initial_state_idx,
        )
    ):
        raise NotImplementedError(
            "Portable GLM5-Next causal_conv1d_update does not support "
            "speculative/APC state rotation"
        )

    if query_start_loc is not None:
        if x.ndim != 2:
            raise ValueError("Varlen causal-conv update expects [tokens, dim]")
        boundaries = query_start_loc.detach().to("cpu", torch.int64).tolist()
        output = torch.empty_like(x)
        for request, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            state_id = int(conv_state_indices[request].item())
            if state_id == null_block_id or state_id < 0:
                output[start:end].zero_()
                continue
            request_output, next_state = _causal_conv_sequence(
                x[start:end].transpose(0, 1).to(conv_state.dtype),
                conv_state[state_id],
                weight,
                bias,
                activation,
            )
            output[start:end] = request_output.transpose(0, 1).to(output.dtype)
            conv_state[state_id].copy_(next_state)
        return output

    squeeze = x.ndim == 2
    x_3d = x.unsqueeze(-1) if squeeze else x
    if x_3d.ndim != 3:
        raise ValueError("Decode causal-conv input must be [batch, dim, tokens]")
    output = torch.empty_like(x_3d)
    for request in range(x_3d.shape[0]):
        state_id = (
            request
            if conv_state_indices is None
            else int(conv_state_indices[request].item())
        )
        if state_id == null_block_id or state_id < 0:
            output[request].zero_()
            continue
        request_output, next_state = _causal_conv_sequence(
            x_3d[request].to(conv_state.dtype),
            conv_state[state_id],
            weight,
            bias,
            activation,
        )
        output[request] = request_output.to(output.dtype)
        conv_state[state_id].copy_(next_state)
    return output.squeeze(-1) if squeeze else output


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform on the last dimension."""
    if x.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"Hadamard input must have dim 128, got {x.shape[-1]}")
    out = x.float()
    width = 1
    while width < INDEX_HEAD_DIM:
        shape = (*out.shape[:-1], -1, 2, width)
        pair = out.reshape(shape)
        left, right = pair.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2).reshape_as(out)
        width *= 2
    return out * (INDEX_HEAD_DIM**-0.5)


def _quantize_fp8_vector(
    x: torch.Tensor, round_scale: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype, fp8_max = get_fp8_dtype_and_max()
    absmax = x.abs().amax(dim=-1).clamp_min(1e-4)
    scale = absmax / fp8_max
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    quant = (x / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    try:
        quant = quant.to(fp8_dtype)
    except RuntimeError as exc:
        raise RuntimeError(
            "The portable GLM5-Next indexer requires float8_e4m3fn support; "
            "provide the FlagGems FP8 cache operators for this platform."
        ) from exc
    return quant, scale.float()


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference Hadamard-128 plus per-row ue8m0 FP8 quantization."""
    if q.ndim != 2 or q.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(f"Expected [rows, 128] query, got {tuple(q.shape)}")
    rotated = hadamard128(q).to(torch.bfloat16).float()
    quantized, scales = _quantize_fp8_vector(rotated, round_scale=True)
    return quantized, scales.unsqueeze(-1)


def _cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3:
        raise ValueError("Indexer cache must be a 3-D uint8 tensor")
    num_blocks, page_size, _ = kv_cache.shape
    flat = kv_cache.view(num_blocks, -1)
    value_bytes = page_size * head_dim
    values = flat[:, :value_bytes].view(num_blocks, page_size, head_dim)
    scales = flat[:, value_bytes:].view(torch.float32).view(num_blocks, page_size, -1)
    return values, scales


def write_fp8_cache(
    kv_cache: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    locations: torch.Tensor,
    head_dim: int,
    write_mask: torch.Tensor | None = None,
) -> None:
    values, cache_scales = _cache_views(kv_cache, head_dim)
    page_size = kv_cache.shape[1]
    loc = locations.to(torch.int64)
    valid = loc >= 0
    if write_mask is not None:
        valid = valid & write_mask.bool()
    if not bool(valid.any()):
        return
    blocks = torch.div(loc[valid], page_size, rounding_mode="floor")
    offsets = torch.remainder(loc[valid], page_size)
    values[blocks, offsets] = quantized[valid].view(torch.uint8)
    scale_values = scales[valid]
    if scale_values.ndim == 1:
        scale_values = scale_values.unsqueeze(-1)
    cache_scales[blocks, offsets, : scale_values.shape[-1]] = scale_values


def kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Exact PyTorch composition of GLM's pool/rotate/FP8 cache write."""
    if slot_k.shape != slot_score.shape:
        raise ValueError("slot_k and slot_score must have identical shapes")
    if slot_k.shape[1:] != (pool_size, head_dim):
        raise ValueError("Unexpected kpool input shape")
    probabilities = torch.softmax(
        slot_score.float() + ape.float().unsqueeze(0), dim=1
    )
    pooled = (slot_k.float() * probabilities).sum(dim=1)
    # This BF16 round-trip is part of the supplied reference implementation.
    rotated = hadamard128(pooled.to(torch.bfloat16).float())
    quantized, scales = _quantize_fp8_vector(rotated, round_scale)
    if write_cache:
        write_fp8_cache(
            kv_cache,
            quantized,
            scales,
            loc,
            head_dim,
            write_mask=write_mask,
        )
    if return_compressed:
        return quantized, scales
    return None


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Ordered portable decode tail update, including pool-completion writes."""
    num_requests, next_n = key.shape[:2]
    for request in range(num_requests):
        for token in range(next_n):
            position = int(positions[request, token].item())
            tail_loc = int(tail_slot_mapping[request, token].item())
            if position < 0 or tail_loc < 0:
                continue
            tail_block = tail_loc // pool_size
            tail_offset = tail_loc % pool_size
            # Stash first.  A completion token must be visible to the subsequent
            # compression read, and tokens of one request are processed in order.
            tail_kv_cache[tail_block, 0, tail_offset].copy_(key[request, token])
            tail_kv_cache[tail_block, 1, tail_offset].copy_(
                slot_score[request, token]
            )
            cache_loc = int(slot_mapping[request, token].item())
            if position % pool_size != pool_size - 1 or cache_loc < 0:
                continue
            kpool_compress_and_write_cache(
                kv_cache,
                tail_kv_cache[tail_block, 0].unsqueeze(0),
                tail_kv_cache[tail_block, 1].unsqueeze(0),
                ape,
                torch.tensor([cache_loc], dtype=torch.int64, device=key.device),
                pool_size,
                head_dim=head_dim,
                round_scale=round_scale,
            )


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed every request's trailing pool in its request-owned tail ring."""
    del head_dim
    num_tokens = tail_slot_mapping.shape[0]
    if num_tokens == 0:
        return
    slots = tail_slot_mapping.to(torch.int64)
    ahead = torch.full_like(slots, -1)
    if num_tokens > kpool:
        ahead[:-kpool] = slots[kpool:]
    valid = slots >= 0
    own_blocks = torch.div(slots.clamp_min(0), kpool, rounding_mode="floor")
    ahead_blocks = torch.div(ahead.clamp_min(0), kpool, rounding_mode="floor")
    same_request = (ahead >= 0) & (ahead_blocks == own_blocks)
    keep = valid & ~same_request
    kept_slots = slots[keep]
    blocks = torch.div(kept_slots, kpool, rounding_mode="floor")
    offsets = torch.remainder(kept_slots, kpool)
    tail_kv_cache[blocks, 0, offsets] = key[keep]
    tail_kv_cache[blocks, 1, offsets] = gate_score[keep]


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if topk % pool_size:
        raise ValueError("topk must be divisible by pool_size")
    offsets = torch.arange(pool_size, device=group_ids.device, dtype=torch.int64)
    token_ids = (group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets).reshape(
        group_ids.shape[0], topk
    )
    valid = group_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape_as(token_ids)
    if page_table is not None:
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, 1, safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        output = (token_ids + topk_offsets.reshape(-1, 1).to(torch.int64)).to(
            torch.int32
        )
    else:
        output = token_ids.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if pool_size == 1:
        return topk_result
    rows, history_len = topk_result.shape
    out_cols = history_len + pool_size - 1
    cols = torch.arange(out_cols, device=topk_result.device).unsqueeze(0)
    is_history = cols < history_len
    safe_hist = cols.clamp(max=history_len - 1).expand(rows, -1)
    history = torch.gather(topk_result, 1, safe_hist)
    tail_start = pool_lens.to(torch.int32) * pool_size
    tail_offset = cols - history_len
    tail_count = seq_lens.to(torch.int32) - tail_start
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count.unsqueeze(1))
    tail_raw = tail_start.unsqueeze(1) + tail_offset
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail = (tail_raw + topk_offsets.reshape(-1, 1).to(torch.int64)).to(
            torch.int32
        )
    else:
        tail = tail_raw.to(torch.int32)
    out = torch.where(is_history, history, torch.full_like(history, -1))
    return torch.where(is_tail, tail, out)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Reference composition used when the fused Triton kernel is unavailable."""
    topk = pool_ids.shape[1] * pool_size
    expanded = expand_pools_to_tokens(
        pool_ids,
        pool_ids >= 0,
        topk,
        pool_size,
    )
    pool_lens = torch.div(seq_lens, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    return append_tail_to_topk(expanded, seq_lens, pool_lens, pool_size)


__all__ = [
    "append_tail_to_topk",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_kda_with_safe_gate",
    "expand_pools_and_append_tail",
    "expand_pools_to_tokens",
    "fwht128_quant_fp8",
    "fused_recurrent_kda",
    "get_fp8_dtype_and_max",
    "hadamard128",
    "kpool_compress_and_write_cache",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "kpool_seed_tail_cache",
    "recurrent_kda",
    "safe_kda_gate",
    "safe_kda_gate_chunk_cumsum",
    "write_fp8_cache",
]
