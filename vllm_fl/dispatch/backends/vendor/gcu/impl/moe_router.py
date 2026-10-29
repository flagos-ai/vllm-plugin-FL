# Copyright (c) 2026 BAAI. All rights reserved.

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _invalidate_padding(ids, padding, count, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(tl.program_id(0) * BLOCK, count, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        valid = offsets < count
        padded = tl.load(padding + offsets // TOPK, mask=valid, other=False)
        tl.store(ids + offsets, -1, mask=valid & padded)


def invalidate_padding_ids(ids: torch.Tensor, padding: torch.Tensor) -> None:
    """Match CUDA topk_softmax's -1 sentinel without native int64 masked_fill_."""
    if (
        ids.ndim != 2
        or not ids.is_contiguous()
        or ids.dtype not in (torch.int32, torch.int64)
        or padding.ndim != 1
        or not padding.is_contiguous()
        or padding.dtype != torch.bool
        or padding.numel() != ids.shape[0]
        or padding.device != ids.device
        or ids.device.type != "gcu"
    ):
        raise ValueError(
            "GCU routing requires contiguous expert IDs and a matching bool row mask"
        )
    if ids.numel() == 0:
        return
    grid = min(triton.cdiv(ids.numel(), 256), 65535)
    _invalidate_padding[(grid,)](
        ids, padding, ids.numel(), TOPK=ids.shape[1], BLOCK=256
    )
