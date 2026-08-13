# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 W8A8 integration for MetaX."""

from __future__ import annotations

import torch

_glm_moe_dsa_metax_active = False


def is_glm_moe_dsa_metax_active() -> bool:
    return _glm_moe_dsa_metax_active


def is_glm_w8a8_int8_moe(quant_method) -> bool:
    if not is_glm_moe_dsa_metax_active():
        return False

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
        CompressedTensorsW8A8Int8MoEMethod,
    )

    return isinstance(quant_method, CompressedTensorsW8A8Int8MoEMethod)


def _patch_mla_prefill() -> None:
    from flash_attn import flash_attn_varlen_func
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends import fa_utils

    def get_scheduler_metadata(*args, **kwargs):
        return None

    def is_available() -> bool:
        return True

    fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
    fa_utils.reshape_and_cache_flash = ops.reshape_and_cache_flash
    fa_utils.get_scheduler_metadata = get_scheduler_metadata
    fa_utils.is_flash_attn_varlen_func_available = is_available


def _quantize_indexer_query_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    assert not column_major_scales and not tma_aligned_scales
    assert x.shape[-1] % group_size == 0 and x.stride(-1) == 1

    if use_ue8m0 is None:
        use_ue8m0 = fp8_utils.is_deep_gemm_e8m0_used()
    if dtype is None:
        dtype = fp8_utils.current_platform.fp8_dtype()

    output = out_q if out_q is not None else torch.empty_like(x, dtype=dtype)
    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        dtype=torch.float32,
        device=x.device,
    )
    fp8_min, fp8_max = fp8_utils.get_fp8_min_max()
    block = fp8_utils.triton.next_power_of_2(group_size)
    groups = x.numel() // group_size
    fp8_utils._per_token_group_quant_fp8[(groups,)](
        x,
        output,
        scales,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
    )
    return output, scales


def _load_int8_indexer_wk(
    name,
    tensor,
    pending,
    params_dict,
    loaded_params,
    pp_missing_layer_names,
) -> bool:
    if "indexer.wk." not in name or "wk_weights" in name:
        return False

    is_weight = name.endswith(".weight") and tensor.dtype == torch.int8
    is_scale = name.endswith(".weight_scale")
    if not is_weight and not is_scale:
        return False

    if any(name.startswith(prefix) for prefix in pp_missing_layer_names):
        return True

    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = pending.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight = (entry["weight"].float() * entry["scale"].float()).to(
        torch.bfloat16
    )
    del pending[layer_prefix]

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight, 0)
    loaded_params.add(fused_name)
    return True


def _patch_model_loader() -> None:
    from vllm.model_executor.models import deepseek_v2

    original_indexer_loader = deepseek_v2._try_load_fp8_indexer_wk

    def load_indexer_wk(*args):
        if _load_int8_indexer_wk(*args):
            return True
        return original_indexer_loader(*args)

    deepseek_v2._try_load_fp8_indexer_wk = load_indexer_wk
    deepseek_v2.per_token_group_quant_fp8 = _quantize_indexer_query_fp8


def _patch_sparse_indexer() -> None:
    from flag_gems.fused import (
        cp_gather_indexer_k_quant_cache,
        indexer_k_quant_and_cache,
    )
    from flag_gems.ops import fp8_mqa_logits, fp8_paged_mqa_logits
    from flag_gems.ops.fp8_mqa_logits import _fp8_mqa_logits_kernel
    from flag_gems.ops.fp8_paged_mqa_logits import fp8_paged_mqa_logits_kernel
    from flag_gems.runtime.backend._metax.fused.top_k_per_row_prefill import (
        top_k_per_row_prefill,
    )
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers import sparse_attn_indexer

    ops.indexer_k_quant_and_cache = indexer_k_quant_and_cache
    ops.cp_gather_indexer_k_quant_cache = cp_gather_indexer_k_quant_cache
    ops.top_k_per_row_prefill = top_k_per_row_prefill
    _fp8_mqa_logits_kernel.fn.configs = [
        config
        for config in _fp8_mqa_logits_kernel.fn.configs
        if config.kwargs["BLOCK_M"] >= 16
    ]
    fp8_paged_mqa_logits_kernel.configs = [
        config
        for config in fp8_paged_mqa_logits_kernel.configs
        if config.kwargs["BLOCK_H"] >= 16
    ]

    def mqa_logits(q, kv, weights, starts, ends, clean_logits=False):
        q = q[0] if isinstance(q, tuple) else q
        return fp8_mqa_logits(q, kv, weights, starts, ends, clean_logits)

    def paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    ):
        q = q[0] if isinstance(q, tuple) else q
        return fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )

    def top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
    ):
        row_ends = seq_lens[:num_rows]
        return top_k_per_row_prefill(
            logits,
            torch.zeros_like(row_ends),
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
        )

    sparse_attn_indexer.fp8_fp4_mqa_logits = mqa_logits
    sparse_attn_indexer.fp8_fp4_paged_mqa_logits = paged_mqa_logits
    sparse_attn_indexer.SparseAttnIndexer.forward_oot = (
        sparse_attn_indexer.SparseAttnIndexer.forward_cuda
    )
    ops.top_k_per_row_decode = top_k_per_row_decode


def _int8_moe_backend_to_kernel_cls(backend):
    from vllm.model_executor.layers.fused_moe.oracle import int8
    from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

    if backend is int8.Int8MoeBackend.TRITON:
        return [TritonExpertsFL]
    raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")


def _make_int8_moe_quant_config(
    w1_scale,
    w2_scale,
    a1_scale=None,
    a2_scale=None,
    w1_bias=None,
    w2_bias=None,
    per_act_token_quant=False,
):
    from vllm.model_executor.layers.fused_moe.config import (
        int8_w8a8_moe_quant_config,
        int8_w8a16_moe_quant_config,
    )

    assert (a1_scale is None) == (a2_scale is None), (
        "a1_scale and a2_scale must both be provided or both be None"
    )

    if a1_scale is None and not per_act_token_quant:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )
    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
    )


def _patch_int8_moe() -> None:
    from vllm.model_executor.layers.fused_moe.oracle import int8

    int8.backend_to_kernel_cls = _int8_moe_backend_to_kernel_cls
    int8.make_int8_moe_quant_config = _make_int8_moe_quant_config


def apply_model_patches() -> None:
    global _glm_moe_dsa_metax_active

    _patch_mla_prefill()
    _patch_model_loader()
    _patch_sparse_indexer()
    _patch_int8_moe()
    _glm_moe_dsa_metax_active = True
