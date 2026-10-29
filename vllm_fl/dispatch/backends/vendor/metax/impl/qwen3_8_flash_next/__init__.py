# SPDX-License-Identifier: Apache-2.0
"""Narrow MetaX guards for the pinned FGV Qwen3.8 operators."""

from __future__ import annotations

from functools import partial, wraps
from importlib import import_module

import torch

from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as portable


def gdn_prefill(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets=None,
    use_qk_l2norm_in_kernel=True,
    core_attn_out=None,
):
    """Adapt FGV's MetaX V-major chunk result to the vLLM layer contract.

    Qwen's upstream fused_post_conv_prep already normalizes Q/K. Keeping its
    FP32 state and gate boundaries avoids the separate high-level FGV wrapper
    that requires state.dtype == q.dtype. This path is prefill eager only.
    """
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Qwen FGV GDN prefill requires eager execution")
    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("Qwen FGV prefill requires pre-normalized Q/K")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Qwen GDN prefill expects [B,T,H,D] Q/K/V")
    if initial_state is None or initial_state.dtype != torch.float32:
        raise ValueError("Qwen GDN prefill requires FP32 initial recurrent state")
    expected = (v.shape[2], v.shape[3], q.shape[3])
    if initial_state.ndim != 4 or initial_state.shape[-3:] != expected:
        raise ValueError("Qwen GDN state must use [N,HV,V,K] layout")
    metax = import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    result = metax.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    output, final_state = result[1], result[3]
    if core_attn_out is not None:
        core_attn_out.reshape(-1)[: output.numel()].copy_(output.reshape(-1))
    return output, final_state


def qsa_select_paged_tokens(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    out=None,
    num_columns=None,
):
    """MetaX compatibility composition: FGV score + FG ATen top-k + expansion.

    The pinned FGV TLE top-k fails FlagTree atomic type verification. Select
    this provider by policy before capture; never retry after kernel execution.
    Top-k ordering is unnecessary, so sorted=False uses FG's ordinary path.
    Short rows use ascending visible blocks, independent of top-k pad order.
    """
    import flag_gems

    block_topk, width = portable._check_selection_geometry(compress_ratio, token_topk)
    count = q.shape[0]
    out = portable._selection_output(count, width, q.device, out)
    if count == 0:
        return out
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if not isinstance(columns, int) or not 0 <= columns <= capacity:
        raise ValueError("QSA score width must fit the allocated cache capacity")
    scores, visible = portable.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=columns,
    )
    indices = torch.full((count, block_topk), -1, dtype=torch.int32, device=q.device)
    if columns:
        selected_count = min(block_topk, columns)
        from .launch_profile import qsa_topk

        _, selected = qsa_topk(
            scores,
            selected_count,
            dim=-1,
            largest=True,
            sorted=False,
        )
        indices[:, :selected_count].copy_(selected.to(torch.int32))
    ascending = torch.arange(block_topk, dtype=torch.int32, device=q.device)
    indices = torch.where(visible[:, None] <= block_topk, ascending, indices)
    return portable.expand_qsa_block_indices(
        indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )


def qwen38_gdn_mixed_qkv(mixed_qkv):
    """Satisfy pinned MetaX FG's contiguous 2D row-gather input contract.

    The model's QKV slice has a larger QKVZ row stride. MetaX index_select's
    2D dim=0 path currently reads rows as tightly packed; materialize only at
    the Qwen mixed-batch boundary before the inherited core gathers tokens.
    """
    return mixed_qkv.contiguous()


def qwen38_ple_gate(gate):
    """Avoid FlagTree's BF16 sign cast without changing the remaining gate math.

    The sign outputs -1/0/1 are exactly representable in BF16. Cast them back
    before multiplication; abs, clamp, sqrt and sigmoid retain the original
    input dtype and operation order under the existing FlagGems context.
    """
    sign = gate.float().sign().to(gate.dtype)
    return torch.sigmoid(sign * gate.abs().clamp_min(1e-6).sqrt())


def qwen38_small_projection(x, weight):
    """Compatibility for BF16 N=1/4 projections using existing FG matrix-vector ops.

    The pinned FG linear dot path does not lower these narrow output tiles on
    MetaX. Each mv accumulates in FP32 and stores BF16 before stacking. Division
    by hc_count stays in the GR caller, after this BF16 rounding boundary.
    """
    import flag_gems

    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("Qwen3.8 small projection requires [...,K] and [N,K]")
    if weight.shape[0] not in (1, 4):
        raise NotImplementedError("Qwen3.8 small projection only supports N=1/4")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise NotImplementedError("Qwen3.8 small projection requires BF16 inputs")
    if x.device != weight.device:
        raise ValueError("Projection input and weight must share a device")
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    columns = [flag_gems.mv(flat, row) for row in weight.unbind(0)]
    projected = torch.stack(columns, dim=-1)
    return projected.reshape(*x.shape[:-1], weight.shape[0])


def register_qwen38_operators(registry, *, is_platform_available):
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    def available(name):
        if not is_platform_available():
            return False
        if name == "qwen38_gdn_mixed_qkv":
            return True
        if name in {"qwen38_small_projection", "qwen38_ple_gate"}:
            import flag_gems

            symbol = "mv" if name == "qwen38_small_projection" else "sign"
            return callable(getattr(flag_gems, symbol, None))
        if name == "qsa_select_paged_tokens":
            import flag_gems

            return portable.is_available("qsa_mqa_paged") and callable(flag_gems.topk)
        return portable.is_available(name)

    def bind(name):
        fn = globals()[name]

        @wraps(fn)
        def call(*args, **kwargs):
            return fn(*args, **kwargs)

        call._is_available = partial(available, name)
        return call

    registry.register_many(
        [
            OpImpl(
                name,
                "vendor.metax.qwen38",
                BackendImplKind.VENDOR,
                bind(name),
                vendor="metax",
                priority=BackendPriority.VENDOR,
            )
            for name in (
                "gdn_prefill",
                "qsa_select_paged_tokens",
                "qwen38_small_projection",
                "qwen38_ple_gate",
                "qwen38_gdn_mixed_qkv",
            )
        ]
    )
