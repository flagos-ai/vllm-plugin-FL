# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon BF16 MoE fusion kernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.triton_utils import tl, triton


GEMM1_BLOCK_M = 128
GEMM1_PHYSICAL_BLOCK_N = 128
GEMM1_BLOCK_K = 128
GEMM1_GROUP_M = 1
GEMM1_NUM_WARPS = 4
GEMM1_NUM_STAGES = 1

REDUCE_PREFILL_TOKENS = 8192
REDUCE_TOP_K = 8
REDUCE_HIDDEN_SIZE = 2048
REDUCE_BLOCK_N = 512
REDUCE_ROWS_PER_PROGRAM = 1
REDUCE_NUM_WARPS = 8
REDUCE_NUM_STAGES = 1


def supports_hygon_bf16_gemm1_silu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor | None,
    config: Mapping[str, Any],
    *,
    top_k: int,
    activation_name: str,
    apply_router_weight_on_input: bool,
    has_quantization: bool,
    has_bias: bool,
    has_expert_map: bool,
    has_lora: bool,
) -> bool:
    """Return whether GEMM1 can use the generic BF16 fused SiLU path."""
    if sorted_token_ids is None or expert_ids is None:
        return False

    if activation_name != "silu":
        return False
    if any(
        (
            apply_router_weight_on_input,
            has_quantization,
            has_bias,
            has_expert_map,
            has_lora,
        )
    ):
        return False
    if activation.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        return False
    if activation.ndim != 2 or weight.ndim != 3 or output.ndim != 2:
        return False
    num_tokens, hidden_size = activation.shape
    num_experts, physical_intermediate_size, weight_hidden_size = weight.shape
    if (
        num_tokens <= 0
        or hidden_size <= 0
        or num_experts <= 0
        or physical_intermediate_size <= 0
        or top_k <= 0
    ):
        return False
    if physical_intermediate_size % 2 != 0:
        return False
    intermediate_size = physical_intermediate_size // 2
    if weight_hidden_size != hidden_size:
        return False
    if tuple(output.shape) != (
        num_tokens * top_k,
        intermediate_size,
    ):
        return False
    if not activation.is_contiguous() or not output.is_contiguous():
        return False
    if weight.stride(-1) != 1:
        return False
    if not sorted_token_ids.is_contiguous() or not expert_ids.is_contiguous():
        return False

    expected_config = {
        "BLOCK_SIZE_M": GEMM1_BLOCK_M,
        "BLOCK_SIZE_N": GEMM1_PHYSICAL_BLOCK_N,
        "BLOCK_SIZE_K": GEMM1_BLOCK_K,
        "GROUP_SIZE_M": GEMM1_GROUP_M,
        "num_warps": GEMM1_NUM_WARPS,
        "num_stages": GEMM1_NUM_STAGES,
    }
    return all(config.get(key) == value for key, value in expected_config.items())


@triton.jit
def _hygon_bf16_gemm1_silu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_routes,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP_K_VALUE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PHYSICAL_BLOCK_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute gate/up columns together and apply SiLU before storing."""
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, PHYSICAL_BLOCK_N)
    if pid >= num_pid_m * num_pid_n:
        return

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    route_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    route_ids = tl.load(sorted_token_ids_ptr + route_offsets).to(tl.int64)
    route_mask = route_ids < num_valid_routes
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    physical_n_offsets = tl.arange(0, PHYSICAL_BLOCK_N).to(tl.int64)
    output_block_n: tl.constexpr = PHYSICAL_BLOCK_N // 2

    paired_output_offsets = pid_n * output_block_n + physical_n_offsets // 2
    weight_n_offsets = paired_output_offsets + (physical_n_offsets % 2) * (N // 2)

    a_ptrs = (
        a_ptr
        + (route_ids[:, None] // TOP_K_VALUE) * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + k_offsets[:, None] * stride_bk
        + weight_n_offsets[None, :] * stride_bn
    )

    accumulator = tl.zeros(
        (BLOCK_SIZE_M, PHYSICAL_BLOCK_N),
        dtype=tl.float32,
    )
    output_column_mask = paired_output_offsets < N // 2
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = k_block * BLOCK_SIZE_K + k_offsets < K
        a = tl.load(
            a_ptrs,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & output_column_mask[None, :],
            other=0.0,
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Match the separate BF16 path, which rounds GEMM output before SiLU.
    accumulator = accumulator.to(tl.bfloat16)
    gate, up = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, output_block_n, 2).split()
    )
    gate = gate / (1.0 + tl.exp2(-(gate * 1.4426950408889634)))
    result = (gate * up).to(tl.bfloat16)

    output_offsets = pid_n * output_block_n + tl.arange(0, output_block_n)
    c_ptrs = (
        c_ptr + route_ids[:, None] * stride_cm + output_offsets[None, :] * stride_cn
    )
    output_mask = route_mask[:, None] & (output_offsets[None, :] < N // 2)
    tl.store(c_ptrs, result, mask=output_mask)


def invoke_hygon_bf16_gemm1_silu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
) -> None:
    """Run the shape-generic Hygon BF16 GEMM1 + SiLU-and-Mul kernel."""
    num_valid_routes = output.shape[0]
    physical_n = weight.shape[1]
    grid = (
        triton.cdiv(sorted_token_ids.numel(), GEMM1_BLOCK_M)
        * triton.cdiv(physical_n, GEMM1_PHYSICAL_BLOCK_N),
    )
    _hygon_bf16_gemm1_silu_kernel[grid](
        activation,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_routes,
        activation.stride(0),
        activation.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        N=physical_n,
        K=activation.shape[1],
        TOP_K_VALUE=top_k,
        BLOCK_SIZE_M=GEMM1_BLOCK_M,
        PHYSICAL_BLOCK_N=GEMM1_PHYSICAL_BLOCK_N,
        BLOCK_SIZE_K=GEMM1_BLOCK_K,
        GROUP_SIZE_M=GEMM1_GROUP_M,
        num_warps=GEMM1_NUM_WARPS,
        num_stages=GEMM1_NUM_STAGES,
    )


def supports_hygon_fixed_topk8_reduce(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether the validated prefill reduction can be used."""
    if input_tensor.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        return False
    if tuple(input_tensor.shape) != (
        REDUCE_PREFILL_TOKENS,
        REDUCE_TOP_K,
        REDUCE_HIDDEN_SIZE,
    ):
        return False
    if tuple(output.shape) != (
        REDUCE_PREFILL_TOKENS,
        REDUCE_HIDDEN_SIZE,
    ):
        return False
    return input_tensor.is_contiguous() and output.is_contiguous()


@triton.jit
def _hygon_fixed_topk8_reduce_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    row_group = tl.program_id(0)
    hidden_block = tl.program_id(1)

    rows = row_group * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = hidden_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < num_tokens) & (cols[None, :] < hidden_size)
    base = (
        input_ptr
        + rows[:, None] * input_stride_token
        + cols[None, :] * input_stride_hidden
    )

    value0 = tl.load(base + 0 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value1 = tl.load(base + 1 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value2 = tl.load(base + 2 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value3 = tl.load(base + 3 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value4 = tl.load(base + 4 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value5 = tl.load(base + 5 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value6 = tl.load(base + 6 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)
    value7 = tl.load(base + 7 * input_stride_topk, mask=mask, other=0.0).to(tl.float32)

    sum01 = value0 + value1
    sum23 = value2 + value3
    sum45 = value4 + value5
    sum67 = value6 + value7
    accumulator = (sum01 + sum23) + (sum45 + sum67)

    output_offsets = (
        rows[:, None] * output_stride_token + cols[None, :] * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, accumulator, mask=mask)


def invoke_hygon_fixed_topk8_reduce(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run the validated fixed-topk=8 BF16 reduction."""
    num_tokens, _, hidden_size = input_tensor.shape
    grid = (
        triton.cdiv(num_tokens, REDUCE_ROWS_PER_PROGRAM),
        triton.cdiv(hidden_size, REDUCE_BLOCK_N),
    )
    _hygon_fixed_topk8_reduce_kernel[grid](
        input_tensor,
        output,
        num_tokens,
        hidden_size,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_N=REDUCE_BLOCK_N,
        ROWS_PER_PROGRAM=REDUCE_ROWS_PER_PROGRAM,
        num_warps=REDUCE_NUM_WARPS,
        num_stages=REDUCE_NUM_STAGES,
    )


def try_hygon_fixed_topk8_reduce(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Run the fixed reduction when supported and report whether it was used."""
    if not supports_hygon_fixed_topk8_reduce(input_tensor, output):
        return False
    invoke_hygon_fixed_topk8_reduce(input_tensor, output)
    return True
