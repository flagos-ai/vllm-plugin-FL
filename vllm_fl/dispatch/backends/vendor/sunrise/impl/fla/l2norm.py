# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU drop-in for ``fla.ops.l2norm.l2norm_fwd``."""

from __future__ import annotations

from typing import Optional

import torch


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Last-axis L2 normalization on PTPU.

    Mirrors FLA's contract (returns a tensor with the same shape; only the
    last dim is normalized). The FLA helper additionally collapses the
    leading dims to a 2D matrix before passing to its Triton kernel — we
    don't need that with PTPU since ``F.normalize`` (and its accelerated
    counterpart) works on the last dim regardless of rank.
    """
    from torch_ptpu.sgl_kernel import ptpu_l2_normalize as _ptpu_l2_normalize

    out = _ptpu_l2_normalize(x, p=2.0, dim=-1, eps=eps)
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out
