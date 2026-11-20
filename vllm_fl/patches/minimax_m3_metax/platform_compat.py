# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Retained MetaX torch.accelerator compatibility."""


from contextlib import suppress


def _patch_torch_accelerator():
    """Complete the metax (MACA) torch.accelerator API.

    The MACA torch fork ships only the device-management subset of
    torch.accelerator and omits the memory-stats subset that vLLM >= 0.24
    calls unconditionally (MemorySnapshot, memory_profiling, weight loaders).
    Bind the torch.cuda equivalents; current_accelerator() == "cuda" on
    metax. The mtgpu allocator reports 0/empty stats, which vllm tolerates.
    """
    import torch

    if not hasattr(torch, "accelerator"):
        return
    accel = torch.accelerator
    if hasattr(accel, "memory_stats"):
        # Full implementation present - nothing to do.
        return

    for _name in (
        "empty_cache",
        "memory_reserved",
        "memory_stats",
        "memory_allocated",
        "max_memory_allocated",
    ):
        if not hasattr(accel, _name) and hasattr(torch.cuda, _name):
            setattr(accel, _name, getattr(torch.cuda, _name))

    if not hasattr(accel, "reset_peak_memory_stats"):
        _cuda_reset = torch.cuda.reset_peak_memory_stats

        def _safe_reset_peak_memory_stats(device=None):
            try:
                _cuda_reset(device)
            except RuntimeError:
                # mtgpu backend may reject an explicit device before the
                # allocator is initialized; the no-arg variant is the fallback.
                # Allocator not initialized at all yet; nothing to reset.
                with suppress(RuntimeError):
                    _cuda_reset()

        accel.reset_peak_memory_stats = _safe_reset_peak_memory_stats
