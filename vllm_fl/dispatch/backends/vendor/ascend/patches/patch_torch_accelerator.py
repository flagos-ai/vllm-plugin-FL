# Copyright (c) 2026 BAAI. All rights reserved.

"""Make PyTorch's generic accelerator memory API safe on Ascend.

PyTorch 2.10 routes several ``torch.accelerator`` memory calls through the
C10 ``DeviceAllocator`` interface.  torch-npu's caching allocator does not
implement that interface, so vLLM 0.24's ``MemorySnapshot`` otherwise fails
before model loading.  Keep the compatibility shim local to FL's Ascend
platform lifecycle and delegate to torch-npu's native memory APIs.
"""

import torch

_PATCH_MARKER = "_fl_ascend_torch_accelerator_patched"
_NPU_MEMORY_API_MAP = {
    "empty_cache": "empty_cache",
    "memory_stats": "memory_stats",
    "memory_reserved": "memory_reserved",
    "reset_peak_memory_stats": "reset_peak_memory_stats",
    "get_memory_info": "mem_get_info",
}


def patch_torch_accelerator() -> None:
    """Redirect vLLM-used accelerator memory APIs to ``torch.npu``.

    The operation is process-local and idempotent.  A partially available NPU
    runtime is rejected before any assignment, because leaving only part of
    vLLM's memory-accounting path redirected would produce misleading cache
    sizing or fail later during profiling.
    """

    npu = getattr(torch, "npu", None)
    accelerator = getattr(torch, "accelerator", None)
    if npu is None:
        raise RuntimeError(
            "FL Ascend requires torch.npu before installing the "
            "torch.accelerator memory compatibility shim"
        )
    if accelerator is None:
        raise RuntimeError(
            "FL Ascend requires torch.accelerator from the matched PyTorch "
            "runtime"
        )

    missing = [
        source_name
        for source_name in _NPU_MEMORY_API_MAP.values()
        if not callable(getattr(npu, source_name, None))
    ]
    if missing:
        raise RuntimeError(
            "FL Ascend torch.npu runtime is missing required memory APIs: "
            + ", ".join(sorted(missing))
        )

    if getattr(accelerator, _PATCH_MARKER, False):
        mismatched = [
            target_name
            for target_name, source_name in _NPU_MEMORY_API_MAP.items()
            if getattr(accelerator, target_name, None)
            is not getattr(npu, source_name)
        ]
        if mismatched:
            raise RuntimeError(
                "FL Ascend torch.accelerator shim was modified after "
                "installation: " + ", ".join(sorted(mismatched))
            )
        return

    for target_name, source_name in _NPU_MEMORY_API_MAP.items():
        setattr(accelerator, target_name, getattr(npu, source_name))
    setattr(accelerator, _PATCH_MARKER, True)
