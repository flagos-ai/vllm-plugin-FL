# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas (LPU) activation operator implementations.

Only the op whose flag_os impl reaches a LPU-incompatible Triton kernel through
a submodule import lives here: ``gelu_and_mul`` resolves the top-level
``_evas/fused`` override explicitly.  ``silu_and_mul`` is served by the flag_os
impl; ``silu_and_mul_with_clamp`` has no vendor copy — upstream added an
LPU-safe flag_os impl.

"""

from __future__ import annotations

import torch


def gelu_and_mul_evas(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    # Resolve via the top-level attribute first: vendor overrides (evas
    # pure-PyTorch, e.g. _evas/fused/gelu_and_mul.py) are patched into the
    # top-level ``flag_gems`` namespace by SpecOpRegistrar -- the fused
    # submodule import would miss them and hit the Triton kernel, which does
    # not compile on LPU.
    import flag_gems as _flag_gems

    gelu_and_mul = getattr(_flag_gems, "gelu_and_mul", None)
    if gelu_and_mul is None:
        from flag_gems.fused import gelu_and_mul  # noqa: PLC0415

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)
