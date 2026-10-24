# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: the Kunlunxin torch build (2.9.0+cu129) exposes a reduced
# ``torch.accelerator`` surface.  vLLM 0.28 calls some of these APIs
# unconditionally -- e.g. ``vllm.utils.mem_utils.MemorySnapshot.measure()``
# calls ``torch.accelerator.get_memory_info(device)`` -- which raises
# ``AttributeError`` at worker ``init_device`` time.
#
# Bind the missing names to their ``torch.cuda`` equivalents.  The Kunlunxin
# runtime presents XPU devices through the CUDA API surface (the platform
# reports ``device_type == "cuda"``), so the mapping is 1:1.
#
# Guard: only names that are *actually absent* on ``torch.accelerator`` are
# bound, so an existing (possibly signature-changed) API is never overridden.
# Same policy as the MetaX/Ascend/MUSA vendor shims.
# --------------------------------------------------------------------------------

import logging

import torch

logger = logging.getLogger(__name__)

# torch.accelerator name -> torch.cuda name.  Names differ where cuda uses a
# different spelling (get_memory_info -> mem_get_info).
_ACCELERATOR_COMPAT = {
    "get_memory_info": "mem_get_info",
    "empty_cache": "empty_cache",
    "memory_allocated": "memory_allocated",
    "max_memory_allocated": "max_memory_allocated",
    "memory_reserved": "memory_reserved",
    "max_memory_reserved": "max_memory_reserved",
    "memory_stats": "memory_stats",
    "reset_peak_memory_stats": "reset_peak_memory_stats",
    "reset_accumulated_memory_stats": "reset_accumulated_memory_stats",
    "synchronize": "synchronize",
    "device_count": "device_count",
    "current_device": "current_device",
    "set_device": "set_device",
}


def apply_accelerator_compat() -> None:
    """Bind torch.accelerator names that are missing on this torch build."""
    patched = []
    for name, cuda_name in _ACCELERATOR_COMPAT.items():
        if hasattr(torch.accelerator, name):
            continue
        impl = getattr(torch.cuda, cuda_name, None)
        if impl is None:
            continue
        setattr(torch.accelerator, name, impl)
        patched.append("%s<-%s" % (name, cuda_name))
    if patched:
        logger.info(
            "kunlunxin accelerator_compat: bound torch.accelerator names from "
            "torch.cuda (%s) on torch %s",
            ", ".join(patched),
            torch.__version__,
        )
