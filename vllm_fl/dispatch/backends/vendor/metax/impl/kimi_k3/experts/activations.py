"""Accepted SiTU materialization and fused kernel paths."""

import torch
from flaggems_vllm.ops.situ_and_mul import situ_and_mul


def situ(x, beta, linear_beta):
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.float(), up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


def _can_fuse_situ_input(x, beta, linear_beta):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4096
        and x.shape[1] in (384, 1536, 2112, 8448)
        and x.is_contiguous()
        and beta == 4.0
        and linear_beta == 25.0
    )


def situ_out(out, x, beta=1.0, linear_beta=-1.0):
    # The FGV out API requires separate input and output storage.
    if (
        _can_fuse_situ_input(x, beta, linear_beta)
        and out.is_contiguous()
        and out.device == x.device
        and out.dtype == x.dtype
        and tuple(out.shape) == (x.shape[0], x.shape[1] // 2)
        and out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    ):
        situ_and_mul(out, x, beta=beta, linear_beta=linear_beta)
        return
    out.copy_(situ(x, beta, linear_beta if linear_beta > 0 else None))
