# SPDX-License-Identifier: Apache-2.0
"""Validate and localize standard-format expert routes in one GPU kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _localize(
    W,
    I,
    E,
    OW,
    OI,
    OV,
    N: tl.constexpr,
    K: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    ES: tl.constexpr,
    LOCAL: tl.constexpr,
    GLOBAL: tl.constexpr,
    MAPPED: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    ok = x < N
    r = x // K
    c = x % K
    ids = tl.load(I + r * IS0 + c * IS1, ok, 0)
    weights = tl.load(W + r * WS0 + c * WS1, ok, 0)
    if MAPPED:
        in_range = (ids >= 0) & (ids < GLOBAL)
        local = tl.load(E + ids.to(tl.int64) * ES, ok & in_range, -1)
        valid = in_range & (local >= 0) & (local < LOCAL)
    else:
        local = ids
        valid = (ids >= 0) & (ids < LOCAL)
    tl.store(OW + x, tl.where(valid, weights, 0), ok)
    tl.store(OI + x, tl.where(valid, local, 0), ok)
    tl.store(OV + x, valid, ok)


def localize_routing(
    topk_weights,
    topk_ids,
    expert_map,
    *,
    local_experts,
    global_experts,
    return_valid_mask=False,
):
    if expert_map is None:
        if global_experts not in (-1, local_experts):
            raise RuntimeError("EP requires original global-to-local expert_map")
    else:
        if expert_map.ndim != 1 or expert_map.numel() != global_experts:
            raise RuntimeError("invalid global-to-local expert_map shape")
    assert (
        topk_weights.ndim == topk_ids.ndim == 2 and topk_weights.shape == topk_ids.shape
    )
    ow = torch.empty(
        topk_weights.shape, device=topk_weights.device, dtype=topk_weights.dtype
    )
    oi = torch.empty(topk_ids.shape, device=topk_ids.device, dtype=topk_ids.dtype)
    ov = torch.empty(topk_ids.shape, device=topk_ids.device, dtype=torch.bool)
    n = topk_ids.numel()
    if n:
        _localize[(triton.cdiv(n, 256),)](
            topk_weights,
            topk_ids,
            expert_map if expert_map is not None else topk_ids,
            ow,
            oi,
            ov,
            n,
            topk_ids.shape[1],
            *topk_weights.stride(),
            *topk_ids.stride(),
            expert_map.stride(0) if expert_map is not None else 1,
            local_experts,
            global_experts,
            expert_map is not None,
            256,
        )
    return (ow, oi, ov) if return_valid_mask else (ow, oi)
