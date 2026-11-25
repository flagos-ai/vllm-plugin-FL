# Copyright (c) 2026 BAAI. All rights reserved.
"""Monkey-patch for torch.randint on LPU.

``torch.randint`` is unsupported by torch_lpu's aten op checker — it
aborts during model ``__init__`` when the default device is ``'lpu'``
and ``randint`` is called without an explicit ``device`` kwarg (e.g.
``DeepseekV4MoE.__init__`` initialising ``tid2eid``).

The fix: when on LPU, create a CPU tensor and move the result to LPU.
This preserves the full randint API while avoiding the LPU op-checker
crash. Only triggers when no explicit ``device`` is passed.
"""

from __future__ import annotations


def apply_lpu_randint_patch() -> None:
    """Wrap ``torch.randint`` to fall back to CPU when default device is LPU.

    Idempotent — safe to call multiple times.
    """
    import torch

    if getattr(torch, "_lpu_randint_patched", False):
        return

    _orig_randint = torch.randint

    def _lpu_randint(*args, **kwargs):
        device = kwargs.get("device", None)
        if device is None:
            # If no explicit device, check if default device is LPU.
            # Avoid calling torch.device('lpu') directly since that may
            # itself trigger LPU op discovery before torch_lpu is loaded.
            from vllm_fl.utils import DeviceInfo
            if DeviceInfo().device_type == "lpu":
                kwargs["device"] = "cpu"
                result = _orig_randint(*args, **kwargs)
                return result.to(torch.device("lpu"))
        return _orig_randint(*args, **kwargs)

    torch.randint = _lpu_randint
    torch._lpu_randint_patched = True  # type: ignore[attr-defined]
