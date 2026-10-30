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
    """Select with the measured non-TLE FGV paths; skip fully visible buckets.

    Small batches use the corrected native per-row selector. Large batches
    retain radix selection because native per-row regresses at M2048/N2048.
    This local VENDOR policy leaves other FG/FGV consumers unchanged.
    """
    block_topk, width = portable._check_selection_geometry(compress_ratio, token_topk)
    count = q.shape[0]
    out = portable._selection_output(count, width, q.device, out)
    if count == 0:
        return out
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if not isinstance(columns, int) or not 0 <= columns <= capacity:
        raise ValueError("QSA score width must fit the allocated cache capacity")
    if columns <= block_topk:
        # This graph bucket fits entirely in the selection budget. Expansion
        # applies device-side visibility, request validity and incomplete tails.
        indices = torch.arange(block_topk, dtype=torch.int32, device=q.device)
        indices = indices.unsqueeze(0).expand(count, -1)
    else:
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
        if block_topk == 512 and columns <= 32768:
            if count <= 128:
                native = import_module("flaggems_vllm.ops.top_k_per_row_decode")
                if native.HAS_TLE:
                    raise RuntimeError(
                        "Qwen QSA requires the validated FGV non-TLE selector; "
                        "disable per-row TLE before worker startup"
                    )
                indices = torch.empty(
                    (count, block_topk), device=scores.device, dtype=torch.int32
                )
                native.top_k_per_row_decode(
                    scores,
                    1,
                    visible,
                    indices,
                    count,
                    scores.stride(0),
                    scores.stride(1),
                    block_topk,
                )
            else:
                select = import_module(
                    "flaggems_vllm.runtime.backend._metax.ops.radix_topk"
                ).radix_topk
                indices = select(scores, visible, block_topk)
        else:
            # Preserve the original provider outside the measured QSA geometry.
            import flag_gems

            _, selected = flag_gems.topk(
                scores, block_topk, dim=-1, largest=True, sorted=False
            )
            ascending = torch.arange(block_topk, dtype=torch.int32, device=q.device)
            indices = torch.where(
                visible[:, None] <= block_topk, ascending, selected.to(torch.int32)
            )
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
    """Select the measured BF16 narrow projection without moving GR rounding.

    The MetaX N=4/K=10240 path uses native mm.dtype with FP32 accumulation,
    followed by the original BF16 store boundary. This explicit overload is
    outside FlagGems' registered linear/mm APIs; a global blacklist cannot
    express this model shape. Other N=4 shapes retain the FG mv fallback.
    The N=1 result is already a contiguous column, so stacking adds only a copy.
    Division by hc_count remains in the GR caller after the BF16 store.
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
    if weight.shape[0] == 1:
        return flag_gems.mv(flat, weight[0]).reshape(*x.shape[:-1], 1)
    if (
        flat.device.type == "cuda"
        and flat.shape[1] == 10240
        and flat.shape[0] <= 2048
        and flat.is_contiguous()
        and weight.is_contiguous()
    ):
        projected = torch.ops.aten.mm.dtype(flat, weight.T, out_dtype=torch.float32).to(
            torch.bfloat16
        )
        return projected.reshape(*x.shape[:-1], 4)
    columns = [flag_gems.mv(flat, row) for row in weight.unbind(0)]
    return torch.stack(columns, dim=-1).reshape(*x.shape[:-1], 4)


def qsa_sparse_split_count(q, k_cache, topk):
    """Use graph-static split-KV only for the measured MetaX QSA bucket."""
    if (
        q.device.type == "cuda"
        and q.dtype == k_cache.dtype == torch.bfloat16
        and 0 < q.shape[0] <= 64
        and q.shape[1:] == (3, 256)
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 256)
        and topk == 2051
    ):
        return 16 if q.shape[0] <= 32 else 8
    return 1


def qwen38_grdown_projection(x, weight):
    """Fixed native path for graph-dominated GR down shapes, with FG fallback.

    FP32 output from the unregistered aten.mm.dtype overload is rounded to
    BF16 before the GR caller divides by hc_count. Warmup and capture select
    the same path. Small-M eager is slower; this is a measured decode policy.
    """
    from .linear_policy import native_grdown, supports_native_grdown

    if supports_native_grdown(x, weight):
        return native_grdown(x, weight)
    return torch.nn.functional.linear(x, weight)


def register_qwen38_operators(registry, *, is_platform_available):
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    def available(name):
        if not is_platform_available():
            return False
        if name == "qwen38_grdown_projection":
            return True
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
                "qsa_sparse_split_count",
                "qwen38_small_projection",
                "qwen38_grdown_projection",
                "qwen38_ple_gate",
                "qwen38_gdn_mixed_qkv",
            )
        ]
    )
