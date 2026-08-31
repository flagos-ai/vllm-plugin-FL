# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon-optimized BF16 MoE GEMM2 kernel for small decode batches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


logger = init_logger(__name__)
MAX_SMALL_DECODE_TOKENS = 128
_active_expert_log_count = 0


@dataclass(frozen=True)
class HygonBf16MoeGemm2Config:
    block_m: int = 16
    block_n: int = 128
    block_k: int = 128
    num_warps: int = 8
    num_stages: int = 1
    workers_per_cu: int = 4

    @classmethod
    def for_small_decode(
        cls, route_config: Mapping[str, Any]
    ) -> "HygonBf16MoeGemm2Config":
        """Build the optimized GEMM2 launch config from the selected MoE tile."""
        defaults = cls()
        return cls(
            block_m=int(route_config.get("BLOCK_SIZE_M", defaults.block_m)),
            block_n=int(route_config.get("BLOCK_SIZE_N", defaults.block_n)),
            block_k=int(route_config.get("BLOCK_SIZE_K", defaults.block_k)),
            num_warps=int(route_config.get("num_warps", defaults.num_warps)),
            num_stages=int(route_config.get("num_stages", defaults.num_stages)),
            workers_per_cu=int(
                route_config.get("workers_per_cu", defaults.workers_per_cu)
            ),
        )


def supports_hygon_bf16_moe_gemm2(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routed_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    top_k: int,
    config: Mapping[str, Any],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    bias: torch.Tensor | None,
) -> bool:
    """Return whether this invocation matches the validated small-M GEMM2."""
    if activation.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        return False
    if activation.ndim != 2 or weight.ndim != 3 or output.ndim != 3:
        return False
    if not 0 < output.shape[0] <= MAX_SMALL_DECODE_TOKENS:
        return False
    if output.shape[1] != 8 or top_k != 1:
        return False
    if tuple(weight.shape) != (256, 2048, 256):
        return False
    if tuple(activation.shape) != (output.shape[0] * 8, 256):
        return False
    if output.shape[2] != 2048:
        return False
    if routed_weights is None or routed_weights.numel() != activation.shape[0]:
        return False
    if sorted_token_ids is None:
        return False
    if any(
        (
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
        )
    ):
        return False
    if bias is not None:
        return False
    return "BLOCK_SIZE_M" in config


def _maybe_log_active_experts(
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_m: int,
) -> None:
    """Log routed expert count for a small number of eager debug calls."""
    global _active_expert_log_count

    if os.getenv("VLLM_FL_DEBUG_MOE_ACTIVE_EXPERTS", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return

    limit = int(os.getenv("VLLM_FL_DEBUG_MOE_ACTIVE_EXPERTS_LIMIT", "20"))
    if _active_expert_log_count >= limit:
        return

    padded_routes = int(num_tokens_post_padded.detach().cpu().item())
    expert_block_count = triton.cdiv(padded_routes, block_m)
    routed_expert_ids = expert_ids[:expert_block_count].detach().cpu().tolist()
    active_experts = len(
        {int(expert_id) for expert_id in routed_expert_ids if expert_id >= 0}
    )
    logger.info(
        "Hygon BF16 MoE GEMM2 routing: active_experts=%d, "
        "expert_blocks=%d, padded_routes=%d",
        active_experts,
        expert_block_count,
        padded_routes,
    )
    _active_expert_log_count += 1


@triton.jit
def _hygon_bf16_moe_gemm2_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    routed_weight_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    route_count,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    mul_routed_weight: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    worker_id = tl.program_id(0)
    worker_count = tl.num_programs(0)
    n_tiles = tl.cdiv(n_size, block_n)
    padded_routes = tl.load(num_tokens_post_padded_ptr)
    expert_blocks = tl.cdiv(padded_routes, block_m)
    total_tiles = expert_blocks * n_tiles

    tile_id = worker_id
    while tile_id < total_tiles:
        expert_block = tile_id // n_tiles
        n_tile = tile_id - expert_block * n_tiles
        expert_id = tl.load(expert_ids_ptr + expert_block)
        valid_expert = expert_id >= 0

        m_offsets = tl.arange(0, block_m)
        n_offsets = n_tile * block_n + tl.arange(0, block_n)
        route_ids = tl.load(sorted_token_ids_ptr + expert_block * block_m + m_offsets)
        valid_rows = (route_ids < route_count) & valid_expert

        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_start in range(0, k_size, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            a_addresses = (
                a_ptr + route_ids[:, None] * stride_am + k_offsets[None, :] * stride_ak
            )
            b_addresses = (
                b_ptr
                + expert_id * stride_be
                + k_offsets[:, None] * stride_bk
                + n_offsets[None, :] * stride_bn
            )
            a = tl.load(
                a_addresses,
                mask=valid_rows[:, None] & (k_offsets[None, :] < k_size),
                other=0.0,
            )
            b = tl.load(
                b_addresses,
                mask=(k_offsets[:, None] < k_size)
                & (n_offsets[None, :] < n_size)
                & valid_expert,
                other=0.0,
            )
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

        if mul_routed_weight:
            routed_weight = tl.load(
                routed_weight_ptr + route_ids,
                mask=valid_rows,
                other=0.0,
            ).to(tl.float32)
            accumulator *= routed_weight[:, None]

        c_addresses = (
            c_ptr + route_ids[:, None] * stride_cm + n_offsets[None, :] * stride_cn
        )
        tl.store(
            c_addresses,
            accumulator.to(tl.bfloat16),
            mask=valid_rows[:, None] & (n_offsets[None, :] < n_size),
        )
        tile_id += worker_count


def invoke_hygon_bf16_moe_gemm2(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routed_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    config: HygonBf16MoeGemm2Config,
) -> None:
    """Run the persistent BF16 GEMM2 kernel on an existing route plan."""
    if activation.shape[1] != weight.shape[2]:
        raise ValueError("activation K and weight K must match")
    if output.shape[-1] != weight.shape[1]:
        raise ValueError("output N and weight N must match")
    if output.numel() // output.shape[-1] != activation.shape[0]:
        raise ValueError("output route count must match activation rows")
    if activation.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("activation and weight K dimensions must be contiguous")
    if output.stride(-1) != 1:
        raise ValueError("output N dimension must be contiguous")
    if not routed_weights.is_contiguous():
        raise ValueError("routed_weights must be contiguous")
    if not sorted_token_ids.is_contiguous() or not expert_ids.is_contiguous():
        raise ValueError("the aligned route plan must be contiguous")

    for name, value in (
        ("block_m", config.block_m),
        ("block_n", config.block_n),
        ("block_k", config.block_k),
    ):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")

    output_2d = output.view(-1, output.shape[-1])
    device_props = torch.cuda.get_device_properties(activation.device)
    cu_count = int(device_props.multi_processor_count)
    n_tiles = triton.cdiv(weight.shape[1], config.block_n)
    max_tiles = expert_ids.numel() * n_tiles
    workers = min(max_tiles, cu_count * config.workers_per_cu)
    workers = max(1, workers)

    _maybe_log_active_experts(
        expert_ids,
        num_tokens_post_padded,
        config.block_m,
    )

    _hygon_bf16_moe_gemm2_kernel[(workers,)](
        activation,
        weight,
        output_2d,
        routed_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        activation.shape[0],
        n_size=weight.shape[1],
        k_size=weight.shape[2],
        stride_am=activation.stride(0),
        stride_ak=activation.stride(1),
        stride_be=weight.stride(0),
        stride_bn=weight.stride(1),
        stride_bk=weight.stride(2),
        stride_cm=output_2d.stride(0),
        stride_cn=output_2d.stride(1),
        mul_routed_weight=mul_routed_weight,
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
