# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU-safe ``F.pad`` (constant-mode) pure-PyTorch fallback.

``torch.nn.functional.pad`` (``nn.ConstantPad*`` shares its LPU backend kernel)
non-deterministically corrupts its output to zeros/garbage on LPU — reproduced
on derived tensors (einsum/cat/rand-op outputs) and sometimes even plain
``rand`` inputs (firmware race); ``.contiguous()`` does NOT help. The only
correct path is a pure-PyTorch constant pad (new_full + slice copy);
non-constant modes (replicate/reflect/circular) delegate to the original —
vLLM production only uses constant padding.

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIG_PAD = F.pad


def _lpu_safe_pad(input, pad, mode="constant", value=None):
    """Drop-in replacement for ``torch.nn.functional.pad`` on LPU.

    ``constant`` mode is reimplemented in pure PyTorch; every other mode and
    malformed ``pad`` tuples fall through to the original implementation.
    """
    if mode != "constant":
        return _ORIG_PAD(input, pad, mode=mode, value=value)
    if len(pad) % 2 != 0 or len(pad) > 2 * input.ndim:
        return _ORIG_PAD(input, pad, mode=mode, value=value)
    if value is None:
        value = 0.0
    elif isinstance(value, torch.Tensor):
        value = value.item()

    x = input
    n = len(pad) // 2
    for i in range(n):
        dim = x.ndim - 1 - i
        left, right = pad[2 * i], pad[2 * i + 1]
        if left == 0 and right == 0:
            continue
        shape = list(x.shape)
        shape[dim] = shape[dim] + left + right
        out = x.new_full(shape, value)
        idx = [slice(None)] * x.ndim
        idx[dim] = slice(left, left + x.shape[dim])
        out[tuple(idx)] = x
        x = out
    return x


def apply_lpu_pad_patch() -> None:
    """Replace ``F.pad`` with the LPU-safe constant-pad implementation."""
    global _ORIG_PAD
    if F.pad is _lpu_safe_pad:
        return
    _ORIG_PAD = F.pad
    F.pad = _lpu_safe_pad
