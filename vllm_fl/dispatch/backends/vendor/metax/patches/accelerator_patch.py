# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Compatibility shims for older Metax PyTorch accelerator APIs."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _patch_accelerator_api() -> None:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return

    for name in (
        "empty_cache",
        "mem_get_info",
        "memory_allocated",
        "max_memory_allocated",
        "reset_peak_memory_stats",
    ):
        if not hasattr(accelerator, name) and hasattr(cuda, name):
            setattr(accelerator, name, getattr(cuda, name))

    if not hasattr(accelerator, "current_device") and hasattr(cuda, "current_device"):
        accelerator.current_device = cuda.current_device

    if not hasattr(accelerator, "set_device") and hasattr(cuda, "set_device"):
        accelerator.set_device = cuda.set_device


try:
    _patch_accelerator_api()
except Exception:
    logger.debug("Failed to patch torch.accelerator compatibility APIs.", exc_info=True)
