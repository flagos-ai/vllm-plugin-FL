# SPDX-License-Identifier: Apache-2.0
"""Narrow MetaX guards for the pinned FGV Qwen3.8 operators."""

from __future__ import annotations

from functools import partial, wraps
from importlib import import_module

import torch

from .qsa_policy import (
    qsa_select_paged_tokens as qsa_select_paged_tokens,
    qsa_sparse_paged_attention as qsa_sparse_paged_attention,
    qsa_sparse_split_count as qsa_sparse_split_count,
)
from .shape_limits import max_prefill_rows
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
    chunk_skew_bucket=None,
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
        chunk_skew_bucket=chunk_skew_bucket,
    )
    output, final_state = result[1], result[3]
    if core_attn_out is not None:
        core_attn_out.reshape(-1)[: output.numel()].copy_(output.reshape(-1))
    return output, final_state


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
        and flat.shape[0] <= max_prefill_rows()
        and flat.is_contiguous()
        and weight.is_contiguous()
    ):
        projected = torch.ops.aten.mm.dtype(flat, weight.T, out_dtype=torch.float32).to(
            torch.bfloat16
        )
        return projected.reshape(*x.shape[:-1], 4)
    columns = [flag_gems.mv(flat, row) for row in weight.unbind(0)]
    return torch.stack(columns, dim=-1).reshape(*x.shape[:-1], 4)


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


def qwen38_grup_projection(x, weight):
    from .mctlass_linear import linear_or_none

    result = linear_or_none(x, weight)
    return torch.nn.functional.linear(x, weight) if result is None else result


def qwen38_hc_grouped_norm(x, weight, groups, eps):
    from .hc_policy import grouped_norm

    return grouped_norm(x, weight, groups, eps)


def qwen38_hc_gate_reduce(logits, normed, groups):
    from .hc_policy import gate_reduce

    return gate_reduce(logits, normed, groups)


def qwen38_hc_inject_combine(logits, block_output, residual, groups):
    from .hc_policy import inject_combine

    return inject_combine(logits, block_output, residual, groups)


# This provider writes every active row or initializes its fallback output.
# functools.wraps in registration carries the capability to the resolved call.
qsa_sparse_paged_attention._qwen38_full_output = True


def qwen38_qsa_rope(rotary_emb, positions, tensor, cache):
    if (
        tensor.device.type != "cuda"
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != 3
        or tensor.shape[-1] != 128
        or int(rotary_emb.rotary_dim) != 64
        or cache.dtype != tensor.dtype
        or positions.ndim not in (1, 2)
    ):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.rotary_bf16 import (
        rotary_embedding_bf16,
    )

    return rotary_embedding_bf16(
        tensor,
        positions,
        cache,
        int(rotary_emb.rotary_dim),
        neox=bool(rotary_emb.is_neox_style),
        sections=getattr(rotary_emb, "mrope_section", None),
        interleaved=bool(getattr(rotary_emb, "mrope_interleaved", False)),
        fp32_compute=bool(rotary_emb.apply_rotary_emb.enable_fp32_compute),
    )


def qwen38_qsa_compress_norm_rope(
    raw_cache,
    raw_table,
    mapping,
    logical_positions,
    slots,
    compressed_cache,
    norm_weight,
    norm_eps,
    ratio,
    rotary_emb,
    rope_cache,
    *,
    norm_is_fgv=False,
):
    if (
        not norm_is_fgv
        or torch.is_grad_enabled()
        or not norm_weight.is_contiguous()
        or raw_cache.device.type != "cuda"
        or raw_cache.dtype != torch.bfloat16
        or raw_cache.shape[-2:] != (1, 128)
        or int(rotary_emb.rotary_dim) != 64
        or not bool(rotary_emb.is_neox_style)
        or ratio != 4
    ):
        return False
    from flaggems_vllm.ops.gemma_rms_norm import gemma_rms_norm
    from flaggems_vllm.ops.qwen4.qsa import qsa_compress_groups_with_ratio
    from flaggems_vllm.runtime.backend._metax.ops.rotary_bf16 import (
        rotary_embedding_bf16_store,
    )

    # Retain the selected Gemma kernel's reduction and rounding exactly.
    pooled, first_positions = qsa_compress_groups_with_ratio(
        raw_cache, raw_table, mapping, logical_positions, slots, ratio, rope_cache
    )
    normalized = gemma_rms_norm(pooled.reshape(-1, 128), norm_weight, norm_eps).reshape(
        -1, 1, 128
    )
    sections = getattr(rotary_emb, "mrope_section", None)
    positions = first_positions.transpose(0, 1) if sections else first_positions[:, 0]
    rotary_embedding_bf16_store(
        normalized,
        positions,
        rotary_emb._match_cos_sin_cache_dtype(normalized),
        64,
        slots,
        compressed_cache,
        neox=True,
        sections=sections,
        interleaved=bool(getattr(rotary_emb, "mrope_interleaved", False)),
        fp32_compute=bool(rotary_emb.apply_rotary_emb.enable_fp32_compute),
    )
    return True


def qwen38_hc_down_activation(x, groups):
    from .hc_policy import down_activation

    return down_activation(x, groups)


def register_qwen38_operators(registry, *, is_platform_available):
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    def available(name):
        if not is_platform_available():
            return False
        if name in {"qwen38_grdown_projection", "qwen38_grup_projection"}:
            return True
        if name in {
            "qwen38_hc_grouped_norm",
            "qwen38_hc_gate_reduce",
            "qwen38_hc_inject_combine",
            "qwen38_hc_down_activation",
            "qwen38_qsa_rope",
            "qwen38_qsa_compress_norm_rope",
        }:
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
                "qsa_sparse_paged_attention",
                "qwen38_small_projection",
                "qwen38_grdown_projection",
                "qwen38_grup_projection",
                "qwen38_ple_gate",
                "qwen38_gdn_mixed_qkv",
                "qwen38_hc_grouped_norm",
                "qwen38_hc_gate_reduce",
                "qwen38_hc_inject_combine",
                "qwen38_hc_down_activation",
                "qwen38_qsa_rope",
                "qwen38_qsa_compress_norm_rope",
            )
        ]
    )
