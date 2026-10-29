# SPDX-License-Identifier: Apache-2.0
"""Graph-capturable MXFP4/BF16 grouped experts for Kimi K3.

Reuse FlagGems-vllm's device-side routing alignment. Custom GEMMs consume
checkpoint-native FP4/E8M0 tiles; no decoded expert-weight tensor is stored.
The two BF16 GEMM boundaries and FP32 SiTU/router arithmetic are explicit.
"""

from flaggems_vllm.ops.situ_and_mul import _situ_and_mul_kernel as _situ

import torch
import triton
import triton.language as tl
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_singleton,
)

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.mxfp4_kernels import _signed_bf16
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.decode_bits import decode_bits


@triton.jit(do_not_specialize=["ROUTES"])
def _grouped_mm(
    X,
    W,
    S,
    Y,
    SORTED,
    EXPERTS,
    TOTAL,
    ROUTES,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    INPUT_TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BIT_DECODE: tl.constexpr = False,
):
    """Every program owns an expert/token block and an output-column block."""
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    total = tl.load(TOTAL)
    if row_block * BM >= total:
        return
    expert = tl.load(EXPERTS + row_block).to(tl.int64)
    if expert < 0 or expert >= E:
        return
    route_offsets = row_block * BM + tl.arange(0, BM)
    routes = tl.load(SORTED + route_offsets).to(tl.int64)
    valid_route = routes < ROUTES
    rows = routes // INPUT_TOPK
    columns = column_block * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + reduction
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            valid_route[:, None] & (k[None, :] < K),
            other=0,
        )
        packed = tl.load(
            W + (expert * N + columns[None, :]) * (K // 2) + k[:, None] // 2,
            (columns[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.int32)
        exponent = tl.load(
            S + (expert * N + columns[None, :]) * (K // 32) + k[:, None] // 32,
            (columns[None, :] < N) & (k[:, None] < K),
            other=127,
        ).to(tl.int32)
        code = (packed >> ((k[:, None] & 1) * 4)) & 15
        if BIT_DECODE:
            b = decode_bits(code, exponent)
        else:
            b = _signed_bf16(code, exponent)
        acc = tl.dot(a, b, acc)
    tl.store(
        Y + routes[:, None] * N + columns[None, :],
        acc.to(tl.bfloat16),
        valid_route[:, None] & (columns[None, :] < N),
    )


@triton.jit
def _weighted_sum(
    ROUTE_OUTPUT,
    WEIGHTS,
    OUTPUT,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_H: tl.constexpr = 0,
    OUTPUT_START: tl.constexpr = 0,
):
    RH: tl.constexpr = H if ROUTE_H == 0 else ROUTE_H
    tl.static_assert(0 <= OUTPUT_START and OUTPUT_START + RH <= H)
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, RT)
    values = tl.load(
        ROUTE_OUTPUT + (token * TOPK + slots[:, None]) * RH + columns[None, :],
        (slots[:, None] < TOPK) & (columns[None, :] < RH),
        0,
    ).to(tl.float32)
    weights = tl.load(WEIGHTS + token * TOPK + slots, slots < TOPK, 0).to(tl.float32)
    result = tl.sum(values * weights[:, None], axis=0)
    tl.store(
        OUTPUT + token * H + OUTPUT_START + columns,
        result.to(tl.bfloat16),
        columns < RH,
    )


def _alignment(ids, block_m, experts):
    if ids.shape[0] == 1:
        return moe_align_block_size_singleton(ids, block_m)
    # Padding is for routing metadata only. It lets the existing vectorized
    # prefix-scan run for E=896; no weight matrix or model expert is added.
    metadata_experts = triton.next_power_of_2(experts)
    return moe_align_block_size_no_tle(ids, block_m, metadata_experts)


def grouped_experts(
    x, w13, s13, w2, s2, topk_weights, topk_ids, *, beta=4.0, linear_beta=25.0
):
    """Run both expert GEMMs, SiTU and ordered top-k output combination on GPU.

    Routing IDs must satisfy the router contract 0 <= id < E. All workspaces
    depend on tensor metadata, not GPU values, so the call can be captured.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("Expected a 2D BF16 activation")
    experts, twice_intermediate, packed_hidden = w13.shape
    tokens, hidden = x.shape
    intermediate = twice_intermediate // 2
    if hidden % 32 or intermediate % 32 or packed_hidden * 2 != hidden:
        raise ValueError("Expected complete group32 MXFP4 tiles")
    if tuple(w2.shape) != (experts, hidden, intermediate // 2):
        raise ValueError("The expert matrix shapes do not form gate/up/down")
    if tuple(s13.shape) != (experts, twice_intermediate, hidden // 32):
        raise ValueError("Invalid gate/up scales")
    if tuple(s2.shape) != (experts, hidden, intermediate // 32):
        raise ValueError("Invalid down scales")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != tokens
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("Expected matching [tokens, topk] routes and weights")
    if topk_weights.dtype != torch.float32 or topk_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Expected FP32 router weights and integer expert IDs")
    if any(t.dtype != torch.uint8 for t in (w13, s13, w2, s2)):
        raise TypeError("Weights and E8M0 scales remain checkpoint-native U8")
    if not all(
        t.is_contiguous() and t.device == x.device
        for t in (x, w13, s13, w2, s2, topk_weights, topk_ids)
    ):
        raise ValueError("Grouped experts require contiguous tensors on one device")
    if tokens == 0:
        return torch.empty_like(x)
    topk = topk_ids.shape[1]
    if not 1 <= topk <= 16:
        raise ValueError("This K3 candidate supports topk 1..16")
    block_m = 16 if tokens < 256 else (32 if tokens < 2048 else 64)
    block_n1 = block_k1 = block_n2 = block_k2 = 32
    sorted_ids, expert_ids, padded = _alignment(topk_ids, block_m, experts)
    routes = tokens * topk
    # Gate/up is dead before stage 2 reuses its storage. No full BF16 weights
    # and no FP32 [tokens, topk, hidden] contribution tensor are materialized.
    scratch = torch.empty(
        routes * max(hidden, twice_intermediate), device=x.device, dtype=torch.bfloat16
    )
    gate_up = scratch[: routes * twice_intermediate].view(routes, twice_intermediate)
    route_output = scratch[: routes * hidden].view(routes, hidden)
    activated = torch.empty(
        (routes, intermediate), device=x.device, dtype=torch.bfloat16
    )
    output = torch.empty_like(x)
    row_blocks = triton.cdiv(sorted_ids.numel(), block_m)
    _grouped_mm[(row_blocks, triton.cdiv(twice_intermediate, block_n1))](
        x,
        w13,
        s13,
        gate_up,
        sorted_ids,
        expert_ids,
        padded,
        routes,
        twice_intermediate,
        hidden,
        experts,
        topk,
        block_m,
        block_n1,
        block_k1,
        BIT_DECODE=False,
        num_warps=4,
        num_stages=1,
    )
    _situ[(triton.cdiv(routes * intermediate, 256),)](
        gate_up,
        activated,
        routes * intermediate,
        intermediate,
        beta,
        linear_beta,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _grouped_mm[(row_blocks, triton.cdiv(hidden, block_n2))](
        activated,
        w2,
        s2,
        route_output,
        sorted_ids,
        expert_ids,
        padded,
        routes,
        hidden,
        intermediate,
        experts,
        1,
        block_m,
        block_n2,
        block_k2,
        BIT_DECODE=False,
        num_warps=4,
        num_stages=1,
    )
    _weighted_sum[(tokens, triton.cdiv(hidden, 256))](
        route_output,
        topk_weights,
        output,
        hidden,
        topk,
        triton.next_power_of_2(topk),
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
