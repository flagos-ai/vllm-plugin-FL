# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Copyright 2026 FlagOS Contributors
"""DSV4 symmetric W8A8 expert GEMMs through MCTlass, without vllm_metax."""

import functools

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.torch_utils import direct_register_custom_op


@functools.lru_cache(maxsize=1)
def _handle():
    import mctlassEx

    return mctlassEx.mctlassExHandleWrapper()


def kernel_m_key(a, b, c, topk):
    return (
        str(a.device),
        str(b.device),
        str(c.device),
        tuple(a.shape),
        tuple(b.shape),
        tuple(c.shape),
        str(a.dtype),
        str(b.dtype),
        str(c.dtype),
        tuple(a.stride()),
        tuple(b.stride()),
        tuple(c.stride()),
        topk,
    )


_KM_CACHE = {}


def _kernel_m(a, b, c, topk):
    # a is ALREADY the real quantized input, b the real INT8 weight.
    # No bf16->int8 query cast, fake small backing storage, tensor or pointer cache.
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("MCTlass W8A8 kernel selection requires INT8 inputs")
    if not c.is_contiguous():
        raise ValueError("MCTlass output must be contiguous")
    key = kernel_m_key(a, b, c, topk)
    if key not in _KM_CACHE:
        value = int(
            _handle().mctlass_fuse_moe_get_kernel_m(
                a,
                b,
                c.view(-1, c.size(-1)),
                topk,
            )
        )
        if value <= 0:
            raise RuntimeError("MCTlass returned an invalid BLOCK_SIZE_M")
        if len(_KM_CACHE) >= 256:
            _KM_CACHE.clear()
        _KM_CACHE[key] = value
    return _KM_CACHE[key]


def _gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    padded: torch.Tensor,
    em: int,
    topk: int,
    multiply: bool,
) -> None:
    if not c.is_contiguous():
        raise ValueError("MCTlass output must be contiguous")
    _handle().mctlass_fuse_moe_gemm(
        a,
        b,
        c.view(-1, c.size(-1)),
        a_scales,
        b_scales,
        weights,
        token_ids,
        expert_ids,
        padded,
        em,
        topk,
        multiply,
        torch.cuda.current_stream().cuda_stream,
    )


def _gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    padded: torch.Tensor,
    em: int,
    topk: int,
    multiply: bool,
) -> None:
    pass


direct_register_custom_op(
    op_name="dsv4_mctlass_moe_gemm",
    op_func=_gemm,
    mutates_args=["c"],
    fake_impl=_gemm_fake,
)


def fused_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    activation,
    apply_router_weight_on_input,
    expert_map,
    quant_config,
):
    if (
        activation != MoEActivation.from_str("silu")
        or apply_router_weight_on_input
        or expert_map is not None
        or hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.int8
        or w2.dtype != torch.int8
        or tuple(w1.shape) != (256, 512, 4096)
        or tuple(w2.shape) != (256, 4096, 256)
        or topk_ids.shape != (hidden_states.shape[0], 6)
        or topk_weights.shape != topk_ids.shape
        or topk_weights.dtype != torch.float32
        or not quant_config.per_act_token_quant
        or any(
            getattr(quant_config, name) is not None
            for name in (
                "w1_zp",
                "w2_zp",
                "a1_scale",
                "a2_scale",
                "block_shape",
                "w1_bias",
                "w2_bias",
            )
        )
    ):
        raise ValueError("Unsupported shape/quantization for DSV4 TP8 MCTlass MoE")
    if not all(
        t.is_contiguous() for t in (hidden_states, w1, w2, topk_ids, topk_weights)
    ):
        raise ValueError("DSV4 MCTlass MoE requires contiguous inputs")
    result = torch.empty_like(hidden_states)
    chunk_size = 16 * 1024
    m = min(hidden_states.shape[0], chunk_size)
    # Stage 1 and stage 2 outputs reuse storage, as in the validated forward.
    scratch = torch.empty(
        m * 6 * 4096, device=hidden_states.device, dtype=torch.bfloat16
    )
    activated = torch.empty(
        (m * 6, 256), device=hidden_states.device, dtype=torch.bfloat16
    )
    for lo in range(0, hidden_states.shape[0], chunk_size):
        x = hidden_states[lo : lo + chunk_size]
        rows = x.shape[0]
        ids = topk_ids[lo : lo + rows]
        weights = topk_weights[lo : lo + rows]
        first = scratch[: rows * 6 * 512].view(rows, 6, 512)
        second = scratch[: rows * 6 * 4096].view(rows, 6, 4096)
        act = activated[: rows * 6]
        qx, sx = moe_kernel_quantize_input(
            A=x,
            A_scale=None,
            quant_dtype=torch.int8,
            per_act_token_quant=True,
            block_shape=None,
            ocp_mx_scheme=None,
        )
        block_m = _kernel_m(qx, w1, first, 6)
        sorted_ids, experts, padded = moe_align_block_size(
            ids,
            block_m,
            256,
            None,
            ignore_invalid_experts=True,
        )
        em1 = sorted_ids.numel()
        if rows < block_m:
            em1 = min(em1, rows * 6 * block_m)
        torch.ops.vllm.dsv4_mctlass_moe_gemm(
            qx,
            w1,
            first,
            sx,
            quant_config.w1_scale,
            weights,
            sorted_ids,
            experts,
            padded,
            em1,
            6,
            False,
        )
        apply_moe_activation(activation, act, first.view(-1, 512))
        qa, sa = moe_kernel_quantize_input(
            A=act,
            A_scale=None,
            quant_dtype=torch.int8,
            per_act_token_quant=True,
            block_shape=None,
            ocp_mx_scheme=None,
        )
        em2 = sorted_ids.numel()
        if qa.shape[0] < block_m:
            em2 = min(em2, qa.shape[0] * block_m)
        # Stage 2 already has M*6 rows: topk=1, routing weights applied once.
        # routed_scaling_factor=1.5 is already included in the router weights.
        torch.ops.vllm.dsv4_mctlass_moe_gemm(
            qa,
            w2,
            second,
            sa,
            quant_config.w2_scale,
            weights,
            sorted_ids,
            experts,
            padded,
            em2,
            1,
            True,
        )
        ops.moe_sum(second, result[lo : lo + rows])
    return result
