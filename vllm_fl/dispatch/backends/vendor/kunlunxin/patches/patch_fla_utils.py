# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin compatibility: prevent torch.xpu.get_device_name crash in FLA utils.

Problem
-------
Upstream ``vllm.model_executor.layers.fla.ops.utils`` at module level:

    mapping = {"xpu": "intel", ...}          # line 132
    is_intel = _check_platform() == "intel"  # line 147
    is_intel_alchemist = is_intel and \
        "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)   # line 149

On Kunlunxin, Triton backend reports "xpu" → maps to "intel" → line 149
calls ``torch.xpu.get_device_name(0)`` → crashes with
"Torch not compiled with XPU enabled".

The preferred upstream fix is ``"xpu": "nvidia"`` (line 132) but we cannot
modify installed packages.

Fix
---
``ensure_fla_compat()`` does three things in one call:

1. Wraps ``torch.xpu.get_device_name`` with try/except (prevents crash)
2. Force-imports ``utils.py`` (so it loads NOW with the safe wrapper)
3. Overwrites ``device_platform = "nvidia"`` and related flags

After this, all subsequent imports that touch ``utils.py`` find it already
in ``sys.modules`` with correct values.
"""

from __future__ import annotations

import sys

_FLA_UTILS_MOD = "vllm.model_executor.layers.fla.ops.utils"
_patched = False


def _fix_platform_vars() -> None:
    """Correct platform flags in fla utils."""
    mod = sys.modules.get(_FLA_UTILS_MOD)
    if mod is None:
        return
    mod.device_platform = "nvidia"
    mod.is_nvidia = True
    mod.is_intel = False
    mod.is_amd = False
    mod.is_intel_alchemist = False
    mod.is_nvidia_hopper = False
    mod.is_tma_supported = False


def ensure_fla_compat() -> None:
    """One-shot fix. Call BEFORE any import that may load fla ops utils.

    Safe to call multiple times (only the first call does real work).
    """
    global _patched
    if _patched:
        return
    _patched = True

    import torch

    # Step 1: wrap torch.xpu.get_device_name so it never crashes
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "get_device_name"):
        _orig = torch.xpu.get_device_name

        def _safe_get_device_name(idx=0):
            try:
                return _orig(idx)
            except Exception:
                return ""

        torch.xpu.get_device_name = _safe_get_device_name

    # Step 2: force-import utils.py NOW (it loads safely with the wrapper)
    try:
        import vllm.model_executor.layers.fla.ops.utils  # noqa: F401
    except Exception:
        pass

    # Step 3: overwrite platform vars → device_platform = "nvidia"
    _fix_platform_vars()


def _patch_xpu_get_device():
    """Apply Kunlunxin FLA utils compatibility patch before any model import.

    Prevents torch.xpu.get_device_name crash in vllm.model_executor.layers.fla.ops.utils,
    which calls get_device_name at module level when Triton reports backend as 'xpu'.

    NOTE: Must NOT call get_platform_name() here — it accesses current_platform which
    is not yet initialized during platform resolve, causing infinite recursion and log
    flooding. Use PlatformFL.vendor_name (class attribute, resolved at class definition
    time) as a safe static check at this stage.
    """
    from vllm_fl.platform import PlatformFL
    if PlatformFL.vendor_name != "kunlunxin":
        return
    try:
        ensure_fla_compat()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to apply XPU get_device_name patch: %s", e
        )
