"""Single-CTA small-route alignment; no CPU route reads and no E-by-E workspace."""

from flaggems_vllm.ops.moe_align_block_size import (
    _moe_align_small_routes_compact as _align_small,
)

import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.grouped_moe import _alignment as baseline_alignment


def align_small(ids, block_m, experts, *, num_warps=4, record=None):
    if (
        ids.ndim != 2
        or ids.shape[1] != 16
        or not (2 <= ids.shape[0] <= 64)
        or block_m != 16
        or experts != 896
    ):
        return baseline_alignment(ids, block_m, experts)
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise ValueError("Expected contiguous integer route IDs")
    r = ids.numel()
    cap = min(experts, r) + (r - min(experts, r)) // block_m
    si = torch.empty(cap * block_m, dtype=torch.int32, device=ids.device)
    ei = torch.empty(cap, dtype=torch.int32, device=ids.device)
    total = torch.empty(1, dtype=torch.int32, device=ids.device)
    obj = _align_small[(1,)](
        ids,
        si,
        ei,
        total,
        r,
        experts,
        block_m,
        triton.next_power_of_2(r),
        triton.next_power_of_2(experts),
        num_warps=num_warps,
    )
    if record is not None:
        record.update(
            n_regs=obj.n_regs,
            n_spills=obj.n_spills,
            shared=obj.metadata.shared,
            sorted_capacity=si.numel(),
            expert_capacity=ei.numel(),
            workspace_bytes=sum(t.numel() * t.element_size() for t in (si, ei, total)),
        )
    return si, ei, total
