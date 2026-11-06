# Copyright (c) 2026 BAAI. All rights reserved.
"""Inverse GPT-J RoPE and LightOp-compatible INT8 quantization for TP8 prefill."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _inverse_rope_quant(O, POS, CS, Q, S, ST: tl.constexpr,
                        SH: tl.constexpr, CS_STRIDE: tl.constexpr):
    token = tl.program_id(0)
    col = tl.arange(0, 4096)
    head, dim = col // 512, col % 512
    nope = tl.load(O + token * ST + head * SH + dim,
                   dim < 448, other=0).to(tl.float32)
    rot = tl.arange(0, 256)
    rot_head, rot_dim = rot // 32, rot % 32
    even = tl.load(O + token * ST + rot_head * SH + 448 + 2 * rot_dim).to(tl.float32)
    odd = tl.load(O + token * ST + rot_head * SH + 449 + 2 * rot_dim).to(tl.float32)
    position = tl.load(POS + token).to(tl.int64)
    cos = tl.load(CS + position * CS_STRIDE + rot_dim).to(tl.float32)
    sin = tl.load(CS + position * CS_STRIDE + rot_dim + 32).to(tl.float32)
    # Preserve the intermediate BF16 store of the original inverse-RoPE kernel.
    rotated_even = (even * cos + odd * sin).to(tl.bfloat16).to(tl.float32)
    rotated_odd = (odd * cos - even * sin).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(
        tl.max(tl.abs(nope), 0),
        tl.max(tl.maximum(tl.abs(rotated_even), tl.abs(rotated_odd)), 0),
    )
    maximum = tl.maximum(maximum, 1e-10)
    inverse_scale = 127 / maximum
    # LightOp uses reciprocal scaling and round-to-nearest-even, not truncation.
    tl.store(Q + token * 4096 + col,
             libdevice.nearbyint(nope * inverse_scale).to(tl.int8), dim < 448)
    tl.store(Q + token * 4096 + rot_head * 512 + 448 + 2 * rot_dim,
             libdevice.nearbyint(rotated_even * inverse_scale).to(tl.int8))
    tl.store(Q + token * 4096 + rot_head * 512 + 449 + 2 * rot_dim,
             libdevice.nearbyint(rotated_odd * inverse_scale).to(tl.int8))
    tl.store(S + token, maximum / 127)


def prefill_inverse_rope_quant(o, positions, cos_sin):
    """Caller validates BF16 [M,8,512], dense features and contiguous positions."""
    rows = o.shape[0]
    quantized = torch.empty((rows, 4096), device=o.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=o.device, dtype=torch.float32)
    # M controls only the launch grid, so mixed iteration lengths reuse code.
    _inverse_rope_quant[(rows,)](
        o, positions, cos_sin, quantized, scales,
        o.stride(0), o.stride(1), cos_sin.stride(0),
        num_warps=4, num_stages=1,
    )
    return quantized, scales
