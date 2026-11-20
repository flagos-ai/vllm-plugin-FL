# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: Kunlunxin P800 supports pinned host memory but NOT Unified Virtual
# Addressing, so a pinned CPU tensor cannot be mapped into the device address
# space.  Verified on the P800 runtime (torch 2.9.0+cu129, libxpurt.so.2):
#
#   cudaHostGetDevicePointer(pinned)      -> rc 1   (cudaErrorInvalidValue)
#   cuMemHostGetDevicePointer_v2(pinned)  -> rc 1
#   xpu_host_register(pinned, n, 0)       -> rc -707
#   xpu_host_alloc(&p, n, 0)              -> rc 0, but the returned host
#                                            pointer still yields no device
#                                            pointer (rc 1), so there is no
#                                            dual-mapped allocation to use.
#
# vLLM decides between a mapped-host buffer and an explicit-copy buffer with
# ``vllm.utils.platform_utils.is_uva_available()``, which is defined as::
#
#     return is_pin_memory_available() or current_platform.is_cpu()
#
# Pinned memory *is* available on P800, so that probe returns True and the V2
# model runner takes the UVA path: ``UvaBuffer.__init__`` calls
# ``get_accelerator_view_from_cpu_tensor`` -> ``torch.ops._C.
# get_cuda_view_from_cpu_tensor``, which an EMPTY vLLM build does not provide.
# Worker ``init_device`` then dies with::
#
#     RuntimeError: vLLM empty builds do not provide a mapped CUDA view of CPU
#     memory; this backend must provide an implementation or avoid mapped CPU
#     buffers
#
# Upstream already ships the correct fallback for exactly this hardware class:
# ``NonUvaBuffer`` keeps the source of truth on the host and copies explicitly
# to a device tensor, and ``StagedWriteTensor`` keeps large cold tensors in
# device memory instead of mapping them.  Both are selected purely by
# ``is_uva_available()``.  So the fix is to report UVA honestly rather than to
# emulate a mapping the hardware cannot do.
#
# Report ``is_uva_available() == False`` while leaving ``is_pin_memory_available``
# untouched, so pinned staging buffers (and their faster async H2D copies) keep
# working everywhere else.
#
# Guard: only the UVA probe is rebound, and the original callable is kept so the
# patch is idempotent and introspectable.  Same policy as the other Kunlunxin
# vendor shims in this package.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_no_uva"

# Modules that bind ``is_uva_available`` by value via ``from ... import``.
# Each needs its own rebind; patching only the defining module would leave
# these stale references pointing at the original probe.
_IMPORTERS = (
    "vllm.v1.worker.gpu.buffer_utils",
    "vllm.model_executor.offloader.uva",
)


def _no_uva() -> bool:
    """Kunlunxin P800 has pinned memory but no host-to-device mapping."""
    return False


def apply_uva_compat() -> None:
    """Report UVA as unavailable so vLLM picks its explicit-copy buffers."""
    import sys

    from vllm.utils import platform_utils

    original = platform_utils.is_uva_available
    if getattr(original, _PATCH_FLAG, False):
        return

    setattr(_no_uva, _PATCH_FLAG, True)
    _no_uva.__wrapped_original__ = original  # type: ignore[attr-defined]

    # ``is_uva_available`` is functools.cache-wrapped and may already have
    # returned True to an earlier caller. Drop that memo before rebinding so
    # nothing keeps serving the stale answer.
    cache_clear = getattr(original, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()

    platform_utils.is_uva_available = _no_uva

    # Only fix up modules already imported. Importing them here would pull
    # triton and the GPU worker stack in far earlier than vLLM intends; a
    # module imported later reads the patched defining module anyway.
    rebound = ["vllm.utils.platform_utils"]
    for module_name in _IMPORTERS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if getattr(module, "is_uva_available", None) is original:
            module.is_uva_available = _no_uva
            rebound.append(module_name)

    logger.info(
        "kunlunxin uva_compat: reporting is_uva_available()=False in %s; "
        "V2 model runner will use NonUvaBuffer explicit copies because P800 "
        "cannot map pinned host memory into the device address space",
        ", ".join(rebound),
    )
