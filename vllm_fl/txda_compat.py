# Copyright (c) 2026 BAAI. All rights reserved.
"""Early TXDA runtime compatibility without importing dispatch backends."""

from functools import wraps


def patch_accelerator_memory_info() -> bool:
    """Use device-wide TXDA memory information instead of allocator pool sizes.

    The reference torch_txda build reports the caching allocator's pool from
    torch.accelerator.get_memory_info(), including (0, 0) before allocation.
    vLLM 0.32 uses this API to size worker memory, which requires the physical
    device's free/total bytes provided by torch.txda.mem_get_info().
    """
    import torch

    try:
        import torch_txda  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_txda":
            raise
        return False

    accelerator = torch.accelerator
    txda = getattr(torch, "txda", None)
    if txda is None or not txda.is_available():
        return False
    if getattr(accelerator.current_accelerator(), "type", None) != "txda":
        return False

    original = accelerator.get_memory_info
    if getattr(original, "_vllm_fl_txda_memory_info", False):
        return True

    from torch.accelerator._utils import _get_device_index

    mem_get_info = txda.mem_get_info

    @wraps(original)
    def get_memory_info(device_index=None, /) -> tuple[int, int]:
        # Retain the original route if a process changes its active backend.
        if getattr(accelerator.current_accelerator(), "type", None) != "txda":
            return original(device_index)
        # Use PyTorch's own argument validation and current-device semantics.
        # In particular, torch.device("cuda:0") is not a TXDA device even on
        # a vendor build that exposes CUDA compatibility aliases.
        index = _get_device_index(device_index, optional=True)
        return mem_get_info(index)

    get_memory_info._vllm_fl_txda_memory_info = True
    accelerator.get_memory_info = get_memory_info
    return True
