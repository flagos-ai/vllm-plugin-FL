# Copyright (c) 2026 BAAI. All rights reserved.

import torch


def patch_accelerator_memory() -> None:
    """Use the NPU allocator APIs with the validated torch_npu 2.10 stack.

    vLLM 0.28 uses torch.accelerator for profiling and cleanup. PyTorch 2.10
    requires a DeviceAllocator for these APIs, but torch_npu 2.10 provides
    its own allocator. The generic APIs exist but raise an internal assert.
    Keep this bridge scoped to that runtime; other torch versions retain
    their native accelerator implementations.
    """
    if torch.__version__.split(".")[:2] != ["2", "10"]:
        return

    for name in (
        "empty_cache",
        "memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
        "reset_peak_memory_stats",
        "reset_accumulated_memory_stats",
    ):
        setattr(torch.accelerator, name, getattr(torch.npu, name))
    torch.accelerator.get_memory_info = torch.npu.mem_get_info
