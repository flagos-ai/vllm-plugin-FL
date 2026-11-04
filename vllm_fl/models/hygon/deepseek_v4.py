# Copyright (c) 2026 BAAI. All rights reserved.

"""Compile-friendly Hygon wrapper for the upstream DeepSeek-V4 AMD model.

The implementation deliberately reuses the upstream model, weight loader,
MoE and ROCm attention modules.  Hygon MHC and the complete attention module
are exposed as opaque ``torch.ops.vllm`` calls because their validated paths
contain AITER, LightOp and dynamic metadata Python control flow.  The decoder
orchestration, norms and MoE remain visible to Dynamo.
"""

from __future__ import annotations

import logging
import os
from types import MethodType

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4ForCausalLM as _UpstreamDeepseekV4ForCausalLM,
)
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4Model as _UpstreamDeepseekV4Model,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)


def _empty_mhc_pre_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    post_mix = torch.empty(
        *outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape, hidden_size, dtype=residual.dtype, device=residual.device
    )
    return post_mix, comb_mix, layer_input


def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch.backends.vendor.hygon.impl.other.mhc import mhc_pre_hygon

    return mhc_pre_hygon(
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
        norm_weight,
        norm_eps,
    )


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
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
    return _empty_mhc_pre_outputs(residual)


def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm_fl.dispatch.backends.vendor.hygon.impl.other.mhc import mhc_post_hygon

    return mhc_post_hygon(x, residual, post_layer_mix, comb_res_mix)


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    del x, post_layer_mix, comb_res_mix
    return torch.empty_like(residual)


def _mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch.backends.vendor.hygon.impl.other.mhc import (
        mhc_fused_post_pre_hygon,
    )

    return mhc_fused_post_pre_hygon(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )


def _mhc_fused_post_pre_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        x,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )
    post_mix, comb_mix, layer_input = _empty_mhc_pre_outputs(residual)
    return torch.empty_like(residual), post_mix, comb_mix, layer_input


def _attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._vllm_fl_hygon_attention_forward(
        positions, hidden_states, None
    )


def _attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, layer_name
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Fine-grained attention ops (2026-09-12): 粗粒度 fl_hygon_dv4_attention 把
# 整个 attention forward 包成 1 个 op, host Python 量大 → rank 偏斜回潮
# (AR >1ms 从 0.1% 回到 7.7%, 吞吐 -18%)。拆成 3 个细粒度 op 与 vllm 源码
# 修改版(T4=4416 tok/s)的架构对齐:
#   1. fl_hygon_dv4_attn_gemm   — 输入 3 路 GEMM
#   2. fl_hygon_dv4_attn_core   — attention_impl + fused_q_kv_rmsnorm
#   3. fl_hygon_dv4_o_proj      — 输出投影
# norm/split 等 tensor 操作留在编译图内, 让 inductor 融合。
# ---------------------------------------------------------------------------


def _attn_gemm(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[layer_name]
    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        layer.attn_gemm_parallel_execute(hidden_states)
    )
    return (
        qr_kv,
        kv_score if kv_score is not None else hidden_states.new_empty((0,)),
        indexer_kv_score
        if indexer_kv_score is not None
        else hidden_states.new_empty((0,)),
        indexer_weights
        if indexer_weights is not None
        else hidden_states.new_empty((0,)),
    )


def _attn_gemm_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[layer_name]
    t = hidden_states.shape[0]
    qr_kv = hidden_states.new_empty((t, layer.q_lora_rank + layer.head_dim))
    if layer.compressor is not None:
        ks = hidden_states.new_empty(
            (t, layer.compressor.fused_wkv_wgate.weight.shape[0]),
            dtype=torch.float32,
        )
    else:
        ks = hidden_states.new_empty((0,))
    if layer.indexer is not None:
        iw = hidden_states.new_empty((t, layer.indexer.weights_proj.weight.shape[0]))
        iks = hidden_states.new_empty(
            (t, layer.indexer.compressor.fused_wkv_wgate.weight.shape[0]),
            dtype=torch.float32,
        )
    else:
        iw = hidden_states.new_empty((0,))
        iks = hidden_states.new_empty((0,))
    return qr_kv, ks, iks, iw


def _attn_core(
    hidden_states: torch.Tensor,
    qr_kv: torch.Tensor,
    kv_score: torch.Tensor | None,
    indexer_kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
    qr, kv = qr_kv.split([layer.q_lora_rank, layer.head_dim], dim=-1)
    qr, kv = fused_q_kv_rmsnorm(
        qr, kv, layer.q_norm.weight.data, layer.kv_norm.weight.data, layer.eps,
    )
    layer.attention_impl(
        hidden_states, qr, kv, kv_score,
        indexer_kv_score, indexer_weights, positions, out,
    )


def _attn_core_fake(
    hidden_states, qr_kv, kv_score,
    indexer_kv_score, indexer_weights, positions,
    out, layer_name,
) -> None:
    return None


def _o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._o_proj(o, positions)


def _o_proj_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return o.new_empty((o.shape[0], layer.hidden_size))


direct_register_custom_op(
    op_name="fl_hygon_dv4_attn_gemm",
    op_func=_attn_gemm,
    mutates_args=[],
    fake_impl=_attn_gemm_fake,
)
direct_register_custom_op(
    op_name="fl_hygon_dv4_attn_core",
    op_func=_attn_core,
    mutates_args=["out"],
    fake_impl=_attn_core_fake,
)
direct_register_custom_op(
    op_name="fl_hygon_dv4_o_proj",
    op_func=_o_proj,
    mutates_args=[],
    fake_impl=_o_proj_fake,
)


direct_register_custom_op(
    op_name="fl_hygon_dv4_mhc_pre",
    op_func=_mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="fl_hygon_dv4_mhc_post",
    op_func=_mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
direct_register_custom_op(
    op_name="fl_hygon_dv4_mhc_fused_post_pre",
    op_func=_mhc_fused_post_pre,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_fake,
)
direct_register_custom_op(
    op_name="fl_hygon_dv4_attention",
    op_func=_attention,
    mutates_args=[],
    fake_impl=_attention_fake,
)


class _MHCPreGraphOp(nn.Module):
    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.fl_hygon_dv4_mhc_pre(
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
            norm_weight,
            norm_eps,
        )


class _MHCPostGraphOp(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.fl_hygon_dv4_mhc_post(
            x, residual, post_layer_mix, comb_res_mix
        )


class _MHCFusedPostPreGraphOp(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.fl_hygon_dv4_mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )


def _attention_forward_graph(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    del llama_4_scaling
    # 3-op 细粒度: attn_gemm(op) → [host: alloc+slice] → attn_core(op,含norm)
    # → [host: slice] → o_proj(op); 分配和切片留在编译图内让 inductor 优化
    qr_kv, kv_score, indexer_kv_score, indexer_weights = (
        torch.ops.vllm.fl_hygon_dv4_attn_gemm(
            hidden_states, self.prefix
        )
    )

    num_tokens = hidden_states.shape[0]
    o_padded = torch.empty(
        (num_tokens, self.padded_heads, self.head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    torch.ops.vllm.fl_hygon_dv4_attn_core(
        hidden_states, qr_kv, kv_score,
        indexer_kv_score, indexer_weights, positions,
        o_padded, self.prefix,
    )
    o = o_padded[:, : self.n_local_heads, :]
    return torch.ops.vllm.fl_hygon_dv4_o_proj(
        o, positions, self.prefix,
    )


def _install_graph_boundaries(layer: nn.Module) -> None:
    layer.mhc_pre = _MHCPreGraphOp()
    layer.mhc_post = _MHCPostGraphOp()
    layer.mhc_fused_post_pre = _MHCFusedPostPreGraphOp()
    from .prefill_mhc import install_prefill_mhc_norm
    install_prefill_mhc_norm(layer)

    attention = layer.attn
    if hasattr(attention, "_vllm_fl_hygon_attention_forward"):
        return

    object.__setattr__(
        attention,
        "_vllm_fl_hygon_attention_forward",
        attention.forward,
    )
    object.__setattr__(
        attention, "forward", MethodType(_attention_forward_graph, attention)
    )


def configure_attention_graph_boundaries(compilation_config) -> None:
    """Split only the dynamic attention core for piecewise capture.

    GEMM and output projection return newly allocated tensors. Keeping those
    allocations inside captured pieces preserves their addresses on replay.
    The core instead writes to ``out``, allocated by the preceding piece.
    Decode-only full graphs need no attention split at all.
    """
    if not compilation_config.cudagraph_mode.has_piecewise_cudagraphs():
        return
    op = "vllm::fl_hygon_dv4_attn_core"
    if op not in compilation_config.splitting_ops:
        compilation_config.splitting_ops.append(op)


@support_torch_compile
class HygonDeepseekV4Model(_UpstreamDeepseekV4Model):
    """Upstream AMD model with Hygon-specific fullgraph boundaries."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        configure_attention_graph_boundaries(vllm_config.compilation_config)
        # upstream 在 ROCm 上禁用 aux streams("hang issues", amd/model.py);
        # HCU 实测无此问题。构造后给每个 attention 注入 3 条 aux stream,
        # 与 VLLM_ROCM_ENABLE_AUX_STREAMS=1 env 门控一致(默认关)。
        aux_stream_list = None
        if (
            os.environ.get("VLLM_ROCM_ENABLE_AUX_STREAMS", "0") == "1"
            and torch.cuda.is_available()
        ):
            aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if aux_stream_list is not None:
                attn = getattr(layer, "attn", None)
                if attn is not None and getattr(attn, "aux_stream_list", None) is None:
                    attn.aux_stream_list = aux_stream_list
            _install_graph_boundaries(layer)
        logger.info(
            "Enabled Hygon DeepSeek-V4 torch.compile boundaries for %d layers"
            " (aux_streams=%s, prefill_mhc_norm_layers=%d)",
            self.end_layer - self.start_layer,
            aux_stream_list is not None,
            sum(hasattr(layer, "_hygon_original_mhc_forward") for layer in self.layers),
        )


class DeepseekV4ForCausalLM(_UpstreamDeepseekV4ForCausalLM):
    """Hygon entry point retaining upstream loading and logits behavior."""

    model_cls = HygonDeepseekV4Model


__all__ = ["DeepseekV4ForCausalLM", "HygonDeepseekV4Model"]
