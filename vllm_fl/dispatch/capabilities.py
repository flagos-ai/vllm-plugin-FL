# SPDX-License-Identifier: Apache-2.0
"""Runtime kernel and public operator capabilities shared by adapters/providers."""

import torch


def get_flaggems_vllm_operator(name: str):
    """Look up a public operator without importing model/attention adapters."""
    try:
        import flaggems_vllm
    except ModuleNotFoundError as exc:
        if exc.name != "flaggems_vllm":
            raise
        return None
    return getattr(flaggems_vllm, name, None)


def has_device_kernel(name: str, device_type: str) -> bool:
    """A schema or a Meta kernel alone does not implement runtime execution."""
    key = {"cuda": "CUDA", "cpu": "CPU", "xpu": "XPU"}.get(device_type, "PrivateUse1")
    try:
        return any(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, k)
            for k in (key, "CompositeImplicitAutograd", "CompositeExplicitAutograd")
        )
    except RuntimeError:
        return False
