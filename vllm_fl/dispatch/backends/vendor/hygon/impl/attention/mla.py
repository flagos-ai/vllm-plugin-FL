# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon compatibility helpers for DeepSeek-V4 ROCm MLA."""

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _VendorInt8Backend:
    name: str
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    gemm: Callable[..., Any]


_vendor_int8_backend: _VendorInt8Backend | None = None
_vendor_int8_backend_checked = False


def _load_vendor_int8_backend() -> _VendorInt8Backend | None:
    """Load the same LightOp/LMSlim W8A8 primitives used by vllm_hcu."""
    global _vendor_int8_backend, _vendor_int8_backend_checked
    if _vendor_int8_backend_checked:
        return _vendor_int8_backend

    _vendor_int8_backend_checked = True
    gemm = None
    gemm_source = None
    try:
        from lightop import gemm_ops

        gemm = gemm_ops.hipblaslt_w8a8_gemm
        gemm_source = "LightOp"
    except Exception as lightop_error:
        logger.debug("LightOp W8A8 GEMM is unavailable: %s", lightop_error)
        try:
            from lmslim import quant_ops

            gemm = quant_ops.hipblaslt_w8a8_gemm
            gemm_source = "LMSlim"
        except Exception as lmslim_error:
            logger.debug("LMSlim W8A8 GEMM is unavailable: %s", lmslim_error)

    quantize = None
    quant_source = None
    try:
        from lightop.quant import per_token_quant_int8

        quantize = per_token_quant_int8
        quant_source = "LightOp"
    except Exception as lightop_error:
        logger.debug("LightOp INT8 quantizer is unavailable: %s", lightop_error)
        try:
            from lmslim.layers.gemm.int8_utils import per_token_quant_int8

            quantize = per_token_quant_int8
            quant_source = "LMSlim"
        except Exception as lmslim_error:
            logger.debug("LMSlim INT8 quantizer is unavailable: %s", lmslim_error)

    if gemm is None or quantize is None:
        logger.warning(
            "LightOp/LMSlim Hygon W8A8 backend is unavailable; DeepSeek-V4 "
            "wo_a will use the BF16 dequantization fallback"
        )
        return None

    sources = gemm_source if gemm_source == quant_source else f"{gemm_source}+{quant_source}"
    _vendor_int8_backend = _VendorInt8Backend(sources, quantize, gemm)
    logger.info("Using %s hipBLASLt W8A8 backend for DeepSeek-V4 wo_a", sources)
    return _vendor_int8_backend


def _logical_int8_weight_nk(
    wo_a: torch.nn.Module,
    out_features: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Return the logical ``[N, K]`` INT8 weight for either kernel layout."""
    weight = wo_a.weight
    expected_nk = (out_features, hidden_dim)
    expected_kn = (hidden_dim, out_features)
    if tuple(weight.shape) == expected_nk:
        return weight
    if tuple(weight.shape) == expected_kn:
        cached = getattr(wo_a, "_dsv4_wo_a_int8_nk", None)
        if cached is None:
            cached = weight.t().contiguous()
            wo_a._dsv4_wo_a_int8_nk = cached
        return cached
    raise ValueError(
        "Unexpected INT8 wo_a weight shape: expected logical [N, K] "
        f"{expected_nk} or Triton [K, N] {expected_kn}, got {tuple(weight.shape)}"
    )


def _logical_weight_scale_n1(
    wo_a: torch.nn.Module,
    out_features: int,
) -> torch.Tensor:
    """Normalize per-tensor/per-channel weight scales for group slicing."""
    weight_scale = wo_a.weight_scale
    if weight_scale.numel() == 1:
        return weight_scale.reshape(1, 1)
    if weight_scale.numel() == out_features:
        return weight_scale.reshape(out_features, 1)
    raise ValueError(
        "INT8 wo_a requires a per-tensor or per-output-channel scale: "
        f"expected 1 or {out_features} values, got {weight_scale.numel()}"
    )


def _apply_vendor_int8_linear(
    input: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    backend: _VendorInt8Backend,
) -> torch.Tensor:
    """Call the vllm_hcu-compatible hipBLASLt W8A8 linear primitive."""
    input_q, input_scale = backend.quantize(input)
    m, k = input_q.shape
    n = weight_nk.shape[0]
    if weight_nk.shape[1] != k:
        raise ValueError(
            "hipblaslt_w8a8_gemm expects weight [N, K], but got "
            f"input K={k}, weight={tuple(weight_nk.shape)}"
        )

    gemm_result = backend.gemm(
        input_q,
        weight_nk,
        input_scale,
        weight_scale,
        m,
        n,
        k,
        "NT",
        out_dtype,
    )
    if not isinstance(gemm_result, tuple) or len(gemm_result) < 2:
        raise RuntimeError(
            f"{backend.name} hipblaslt_w8a8_gemm returned an unexpected result"
        )
    return gemm_result[1]


def _int8_wo_a_group_gemm(
    o_ref: torch.Tensor,
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    backend: _VendorInt8Backend,
) -> torch.Tensor:
    """Run the block-diagonal ``wo_a`` projection as one INT8 GEMM per group."""
    num_tokens = o_ref.shape[0]
    hidden_dim = o_ref.shape[-1]
    out_features = n_local_groups * o_lora_rank
    weight_nk = _logical_int8_weight_nk(wo_a, out_features, hidden_dim)
    weight_scale = _logical_weight_scale_n1(wo_a, out_features)

    output = torch.empty(
        (num_tokens, n_local_groups, o_lora_rank),
        device=o_ref.device,
        dtype=torch.bfloat16,
    )
    for group_idx in range(n_local_groups):
        channel_slice = slice(
            group_idx * o_lora_rank, (group_idx + 1) * o_lora_rank
        )
        group_scale = (
            weight_scale
            if weight_scale.shape[0] == 1
            else weight_scale[channel_slice]
        )
        output[:, group_idx, :] = _apply_vendor_int8_linear(
            o_ref[:, group_idx, :],
            weight_nk[channel_slice],
            group_scale,
            torch.bfloat16,
            backend,
        )
    return output


def _dequantize_int8_wo_a(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    """BF16 fallback when neither LightOp nor LMSlim W8A8 GEMM is available."""
    out_features = n_local_groups * o_lora_rank
    weight_nk = _logical_int8_weight_nk(wo_a, out_features, hidden_dim)
    weight_scale = _logical_weight_scale_n1(wo_a, out_features)
    weight_bf16 = weight_nk.to(torch.float32) * weight_scale.to(torch.float32)
    return weight_bf16.to(torch.bfloat16).reshape(
        n_local_groups, o_lora_rank, hidden_dim
    )


def _patch_int8_wo_a_weight_layout() -> None:
    """Keep BMM INT8 weights in the ``[N, K]`` layout required by hipBLASLt."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_int8 as int8_scheme,
    )

    scheme_cls = int8_scheme.CompressedTensorsW8A8Int8
    original = scheme_cls.process_weights_after_loading
    if getattr(original, "_vllm_fl_hygon_wo_a_layout", False):
        return

    @functools.wraps(original)
    def _process_weights_after_loading_hygon(self, layer: torch.nn.Module) -> None:
        if (
            getattr(layer, "is_bmm", False)
            and layer.weight.dtype == torch.int8
            and not self.is_static_input_scheme
            and self.input_symmetric
        ):
            layer.weight.data = layer.weight.data.contiguous()
            layer.weight_scale.data = layer.weight_scale.data.contiguous()
            layer._vllm_fl_hygon_wo_a_nk = True
            return
        original(self, layer)

    _process_weights_after_loading_hygon._vllm_fl_hygon_wo_a_layout = True
    scheme_cls.process_weights_after_loading = _process_weights_after_loading_hygon


def _patch_rocm_wo_a_bf16_fallback() -> None:
    """Add INT8 awareness to upstream's cached BF16 fallback helper."""
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as mla_sparse

    original = mla_sparse._get_cached_wo_a_bf16
    if getattr(original, "_vllm_fl_hygon_int8", False):
        return

    @functools.wraps(original)
    def _get_cached_wo_a_bf16_hygon(
        wo_a: torch.nn.Module,
        n_local_groups: int,
        o_lora_rank: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
        if cached is not None:
            return cached
        if wo_a.weight.dtype == torch.int8 and hasattr(wo_a, "weight_scale"):
            cached = _dequantize_int8_wo_a(
                wo_a, n_local_groups, o_lora_rank, hidden_dim
            )
            wo_a._dsv4_wo_a_bf16 = cached
            return cached
        return original(wo_a, n_local_groups, o_lora_rank, hidden_dim)

    _get_cached_wo_a_bf16_hygon._vllm_fl_hygon_int8 = True
    mla_sparse._get_cached_wo_a_bf16 = _get_cached_wo_a_bf16_hygon


def _patch_rocm_wo_a_group_gemm() -> None:
    """Route INT8 ``wo_a`` through the vllm_hcu group-GEMM design."""
    import vllm.models.deepseek_v4.amd.rocm as amd_rocm
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as mla_sparse

    attention_cls = amd_rocm.DeepseekV4ROCMAiterMLAAttention
    original = attention_cls._o_proj
    if getattr(original, "_vllm_fl_hygon_int8_group_gemm", False):
        return

    @functools.wraps(original)
    def _o_proj_hygon(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype != torch.int8:
            return original(self, o, positions)

        backend = _load_vendor_int8_backend()
        if backend is None:
            return original(self, o, positions)

        o_ref = mla_sparse._fused_inverse_rope_gptj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.rope_head_dim,
        )
        o_ref = o_ref.reshape(o.shape[0], self.n_local_groups, -1)
        z = _int8_wo_a_group_gemm(
            o_ref,
            self.wo_a,
            self.n_local_groups,
            self.o_lora_rank,
            backend,
        )
        return self.wo_b(z.flatten(1))

    _o_proj_hygon._vllm_fl_hygon_int8_group_gemm = True
    attention_cls._o_proj = _o_proj_hygon


def patch_rocm_wo_a_int8_group_gemm() -> None:
    """Install Hygon's INT8 group GEMM with a correctness-first BF16 fallback."""
    _patch_int8_wo_a_weight_layout()
    _patch_rocm_wo_a_bf16_fallback()
    _patch_rocm_wo_a_group_gemm()
    logger.info("Patched DeepSeek-V4 ROCm wo_a INT8 group GEMM for Hygon")
