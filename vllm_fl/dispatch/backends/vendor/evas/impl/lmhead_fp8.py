# Copyright (c) 2026 BAAI. All rights reserved.
"""fp8 head backend for the lm_head fp8 patch (``patches/lmhead_fp8.py``).

Contract fixed by the patch: ``build(w) -> (w_fp8, scales)`` and
``matmul(x2d, w_fp8, scales) -> x2d @ w_fp8.T`` in bf16, both pure functions
of their inputs.  Semantics are the production W8A8 ones: 128x128 blocks with
``scale = amax / 448`` fp32 and E4M3 bytes via ``_encode_e4m3`` (no integer
round; see ``_e4m3.py``).  A failed ``kernel_supported`` predicate dequants and
contracts in bf16 instead of raising, so ``matmul`` stays total.

"""

from __future__ import annotations

import logging

import torch

# Weight/activation encoders are platform-local (vendor/evas/local/, 02doc
# §12.5); the old ``flag_gems.fused._e4m3`` fallback was a dead path -- core
# has no such module. The probe covers only the flag_gems op surface: when it
# is missing, the fp8 lm_head keeps W8A16 via ``kernel_supported() -> False``.
from ..local.e4m3 import _encode_e4m3, quant_act_per_block

try:
    from flag_gems import w8a8_block_fp8_matmul
    # kernel_supported has no public binding, so this probe needs the evas
    # path; a FlagGems without _evas lands in the handler below and the
    # fp8 GEMM keeps W8A16.
    from flag_gems.runtime.backend._evas.ops.w8a8_block_fp8_matmul import (
        kernel_supported,
    )
except ImportError:  # pragma: no cover
    from flag_gems import w8a8_block_fp8_matmul

    def kernel_supported(*_args) -> bool:
        return False

logger = logging.getLogger(__name__)

_BLOCK = 128
_FP8_MAX = 448.0
# quant_act_per_block floors the group amax at the same value, so an
# all-zero block cannot divide by zero on either side.
_EPS = 1e-10

__all__ = ["build", "matmul"]

# Rows per build chunk: each fp32 temporary inside the chunk is then
# chunk * k * 4 bytes (16384 x 5120 = 335 MB).  Quantizing the whole
# [248320, 5120] head at once peaked at ~16 GiB of fp32 intermediates and
# OOMed the 48 GiB device -- the fp32 cast, the divide and the e4m3 encoder
# each materialise a full-size tensor.
_CHUNK_ROWS = 128 * _BLOCK


def build(linear_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a bf16 linear weight to uint8 E4M3 + fp32 block scales.

    One scale per 128x128 block (``amax / 448``, fp32, no power-of-two
    rounding) -- the layout ``kernel_supported`` and the vendored kernel
    assert, and the same arithmetic the activation side applies, so weight
    and activation scales are interchangeable in the kernel contract.
    Raises on anything but a 2-D bf16 weight with both dims divisible by
    128 instead of silently taking a ragged layout.  Runs in row chunks to
    bound the fp32 temporaries (see ``_CHUNK_ROWS``); the arithmetic is
    block-local so chunking cannot change the result.
    """
    w = linear_weight
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise ValueError(
            f"lmhead fp8 build: need a 2-D bf16 weight, got "
            f"{tuple(w.shape)} {w.dtype}"
        )
    n, k = w.shape
    if n % _BLOCK or k % _BLOCK:
        raise ValueError(
            f"lmhead fp8 build: shape {tuple(w.shape)} not divisible by "
            f"block {_BLOCK}"
        )
    w_u8 = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    scales = torch.empty(
        (n // _BLOCK, k // _BLOCK), dtype=torch.float32, device=w.device
    )
    for n0 in range(0, n, _CHUNK_ROWS):
        n1 = min(n0 + _CHUNK_ROWS, n)
        wf = w[n0:n1].to(torch.float32).view(
            (n1 - n0) // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK
        )
        amax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp_min(_EPS)
        scale = amax / _FP8_MAX  # [rows//128, 1, k//128, 1] fp32
        wf.div_(scale).clamp_(-_FP8_MAX, _FP8_MAX)
        w_u8[n0:n1] = _encode_e4m3(wf.reshape(n1 - n0, k))
        scales[n0 // _BLOCK : n1 // _BLOCK] = scale.reshape(
            (n1 - n0) // _BLOCK, k // _BLOCK
        )
    return w_u8, scales


# Announced per (rows, k), not once per process: prefill and decode are
# different shapes, and kernel_supported is shape dependent. A single flag
# would let a prefill call print "kernel" while every decode step took the
# fallback -- exactly the silent regression the line exists to catch.
_path_announced: set[tuple[int, int]] = set()


def matmul(x2d: torch.Tensor, w_fp8: torch.Tensor,
           scales: torch.Tensor) -> torch.Tensor:
    """Quantize ``x2d`` per (token, 128-block) and run the block-fp8 GEMM.

    The first call logs which path the GEMM took (``kernel`` or the
    ``dequant-fallback`` the flag_gems op does internally).  The fallback is
    correct but slow; an A/B run must show the ``kernel`` line, otherwise it
    silently measured bf16-contraction speed while reporting fp8.
    """
    x = x2d if x2d.is_contiguous() else x2d.contiguous()
    x_u8, x_s = quant_act_per_block(x, _BLOCK)
    shape = (int(x.shape[0]), int(x.shape[1]))
    if shape not in _path_announced:
        _path_announced.add(shape)
        path = (
            "kernel"
            if kernel_supported(x_u8, w_fp8, x_s, scales, [_BLOCK, _BLOCK])
            else "dequant-fallback"
        )
        logger.info("[lmhead-fp8-impl] matmul path: %s (rows=%d, k=%d)",
                    path, shape[0], shape[1])
    return w8a8_block_fp8_matmul(
        x_u8, w_fp8, x_s, scales, [_BLOCK, _BLOCK],
        output_dtype=torch.bfloat16,
    )
