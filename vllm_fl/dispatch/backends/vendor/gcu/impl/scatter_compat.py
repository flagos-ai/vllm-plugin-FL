# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU workaround for the ``torch.scatter`` family with int64 indices.

torch_gcu's native ``scatter`` / ``scatter_`` / ``scatter_add_`` silently write
only ``ceil(numel / 2)`` elements when the index tensor is int64 -- measured on
torch_gcu 2.11.0 / topsaten 3.8.20260710:

    n = 4096
    torch.zeros(n, dtype=torch.int32, device="gcu").scatter_(
        0, torch.arange(n, dtype=torch.int64, device="gcu"),
        torch.full((n,), 7, dtype=torch.int32, device="gcu"))
    # [0:2048] are 7, [2048:] are still 0
    # n = 1 -> 1, n = 3 -> 2, n = 1025 -> 513 elements written: ceil(n / 2)

The same statement with an int32 index is exact.  Nothing is raised either way;
the kernel simply stops halfway.

``ModelRunnerFL._prepare_input_ids`` writes each continuing request's sampled
token back with

    self.input_ids.gpu.scatter_(dim=0, index=<int64 tensor>, src=<...>)

so on the vendor path only the first half of a batch gets its token back and the
remaining requests decode from a stale one -- their answers repeat or truncate
after the first token, while one-token answers stay correct.  That is exactly
the adaptation gate's ``mixed_concurrent_8`` signature (image-01..03 broken,
the text tasks and image-04 fine); the CPU-side sampler detour in
``impl/sampler.py`` hides the same construct inside the sampler.

The op is wrong for every caller, not just that one, so the tensor methods are
wrapped here (rather than rewriting call sites in the shared runner): an int64
index on a GCU tensor is narrowed to int32 first.  Scatter indices are element
offsets, orders of magnitude below 2**31, so the narrowing is lossless.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _narrow(index):
    """int64 index -> int32, for GCU tensors only (see the module docstring)."""
    if isinstance(index, torch.Tensor) and index.dtype == torch.int64:
        return index.to(torch.int32)
    return index


def _wrap_tensor_method(name: str) -> bool:
    original = getattr(torch.Tensor, name, None)
    if original is None or getattr(original, "_gcu_int32_index", False):
        return False

    def wrapper(self, dim, index, *args, **kwargs):
        if self.device.type == "gcu":
            index = _narrow(index)
        return original(self, dim, index, *args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = "Tensor.%s" % name
    wrapper.__doc__ = (original.__doc__ or "") + (
        "\n\nGCU: int64 indices are narrowed to int32 (torch_gcu native kernels "
        "write only half of the elements otherwise)."
    )
    wrapper._gcu_int32_index = True
    wrapper._gcu_original = original
    setattr(torch.Tensor, name, wrapper)
    return True


def _wrap_torch_function(name: str) -> bool:
    original = getattr(torch, name, None)
    if original is None or getattr(original, "_gcu_int32_index", False):
        return False

    def wrapper(input, dim, index, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "gcu":
            index = _narrow(index)
        return original(input, dim, index, *args, **kwargs)

    wrapper.__name__ = name
    wrapper._gcu_int32_index = True
    setattr(torch, name, wrapper)
    return True


def apply_scatter_int32_patches() -> None:
    """Narrow int64 scatter indices for GCU tensors. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    methods = [n for n in ("scatter", "scatter_", "scatter_add_")
               if _wrap_tensor_method(n)]
    functions = [n for n in ("scatter", "scatter_add") if _wrap_torch_function(n)]
    logger.info(
        "GCU scatter int64-index workaround applied to Tensor.%s and torch.%s",
        ", Tensor.".join(methods) or "(none)", ", torch.".join(functions) or "(none)",
    )
    _PATCHED = True


__all__ = ["apply_scatter_int32_patches"]
