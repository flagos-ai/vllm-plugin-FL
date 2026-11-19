"""Accepted small-M split-K launch; shape policy lives in the layer interface."""
import torch
from .projections import _PARTIAL, _REDUCE


def run(x, w, split, stages):
    partial = torch.empty((split, 8, 1536), device=x.device, dtype=torch.float32)
    out = torch.empty((8, 1536), device=x.device, dtype=torch.bfloat16)
    _PARTIAL[(24, split)](
        x,
        w.T,
        partial,
        8,
        1536,
        7168,
        7168,
        1,
        1,
        7168,
        SPLIT_K=split,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        pipeline="null",
        num_warps=4,
        num_stages=stages,
        num_ctas=1,
    )
    _REDUCE[(48,)](
        partial,
        out,
        8,
        1536,
        1536,
        1,
        SPLIT_K=split,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out, None
