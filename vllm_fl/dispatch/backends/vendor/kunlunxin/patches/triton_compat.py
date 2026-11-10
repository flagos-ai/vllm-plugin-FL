# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: Triton's XPU backend (3.6.0) defines ``XPUOptions`` without
# ``launch_cooperative_grid``.  Triton formats that attribute unconditionally
# when the NVSHMEM post-compile hook is installed, so the first Triton kernel
# compiled by vLLM 0.28 aborts with:
#
#   AttributeError: 'XPUOptions' object has no attribute
#   'launch_cooperative_grid'
#
# vLLM 0.28 zeroes freshly allocated KV blocks through
# ``vllm.v1.worker.utils.KVBlockZeroer`` (a Triton kernel) on the first
# request, so on P800 both offline generation and serving failed while
# initializing that kernel.
#
# Bind the CUDA-compatible default on the class.  ``XPUOptions`` is a frozen
# dataclass; a class-level default is resolved by instance attribute lookup
# and does not change any option the XPU backend actually consumes.  Guard so
# a backend that already provides the attribute is never overridden.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


def apply_triton_compat() -> None:
    """Bind Triton XPU options that the post-compile hook expects."""
    try:
        from triton.backends.xpu.compiler import XPUOptions
    except Exception:
        return

    patched = []
    if not hasattr(XPUOptions, "launch_cooperative_grid"):
        XPUOptions.launch_cooperative_grid = False
        patched.append("launch_cooperative_grid=False")

    if patched:
        logger.info(
            "kunlunxin triton_compat: bound XPUOptions defaults (%s)",
            ", ".join(patched),
        )
