# Copyright (c) 2026 BAAI. All rights reserved.

import logging
from functools import wraps

import torch

logger = logging.getLogger(__name__)

_MEMORY_APIS = {
    "empty_cache": "empty_cache",
    "memory_stats": "memory_stats",
    "memory_allocated": "memory_allocated",
    "memory_reserved": "memory_reserved",
    "max_memory_allocated": "max_memory_allocated",
    "max_memory_reserved": "max_memory_reserved",
    "reset_peak_memory_stats": "reset_peak_memory_stats",
    "reset_accumulated_memory_stats": "reset_accumulated_memory_stats",
    "get_memory_info": "mem_get_info",
}
_MISSING = object()


def _is_our_bridge(function) -> bool:
    """Recognize our bridge even below another owner's functools wrapper."""
    while function is not None:
        if getattr(function, "_vllm_fl_ascend_memory_bridge", False):
            return True
        function = getattr(function, "__wrapped__", None)
    return False


def _make_bridge(name, original, npu_function):
    # Native torch.accelerator functions use the allocator that is absent on
    # the affected stack. A third-party wrapper is retained and called first;
    # only its allocator failure falls back to torch.npu.
    native = getattr(original, "__module__", None) == "torch.accelerator"

    @wraps(original if original is not _MISSING else npu_function)
    def bridge(*args, **kwargs):
        if original is not _MISSING and not native:
            try:
                result = original(*args, **kwargs)
                if name != "memory_stats" or result:
                    return result
            except (AssertionError, NotImplementedError) as exc:
                logger.debug(
                    "Accelerator %s is unavailable; using torch.npu: %s", name, exc
                )
            except RuntimeError as exc:
                if "allocator" not in str(exc).lower():
                    raise
        return npu_function(*args, **kwargs)

    bridge._vllm_fl_ascend_memory_bridge = True
    return bridge


def patch_accelerator_memory() -> bool:
    """Bridge a broken NPU allocator API without leaving a partial install.

    The torch 2.10/torch_npu 2.10 stack can expose generic accelerator
    methods even though their DeviceAllocator is not initialized. Probe the
    actual NPU allocator path before installing this Ascend-only bridge.
    """
    if torch.__version__.split(".")[:2] != ["2", "10"]:
        return False

    try:
        accelerator = torch.accelerator
        npu = torch.npu
        if accelerator.current_accelerator().type != "npu":
            return False

        if all(
            _is_our_bridge(getattr(accelerator, name, None)) for name in _MEMORY_APIS
        ):
            return True

        allocator_probe = getattr(
            getattr(torch, "_C", None), "_accelerator_isAllocatorInitialized", None
        )
        try:
            allocator_ready = allocator_probe() if callable(allocator_probe) else True
        except (AssertionError, RuntimeError):
            allocator_ready = False
        try:
            info = accelerator.get_memory_info()
            accelerator.memory_allocated()
        except (AssertionError, AttributeError, NotImplementedError, RuntimeError):
            pass
        else:
            if (
                allocator_ready
                and isinstance(info, tuple)
                and len(info) == 2
                and info[1] > 0
            ):
                return False

        # Resolve and validate the entire contract before changing any global
        # attribute. Keep each actual original object, including foreign
        # wrappers, so a failed assignment restores its owner exactly.
        npu_functions = {
            name: getattr(npu, npu_name) for name, npu_name in _MEMORY_APIS.items()
        }
        if not all(callable(function) for function in npu_functions.values()):
            raise TypeError("An NPU memory API is not callable")
        originals = {
            name: getattr(accelerator, name, _MISSING) for name in _MEMORY_APIS
        }
        local_originals = {
            name: vars(accelerator).get(name, _MISSING) for name in _MEMORY_APIS
        }
        replacements = {
            name: _make_bridge(name, originals[name], npu_functions[name])
            for name in _MEMORY_APIS
        }

        try:
            for name, replacement in replacements.items():
                setattr(accelerator, name, replacement)
        except Exception:
            namespace = vars(accelerator)
            for name, original in local_originals.items():
                if original is _MISSING:
                    namespace.pop(name, None)
                else:
                    namespace[name] = original
            raise
        return True
    except Exception as exc:
        logger.warning("Failed to bridge Ascend accelerator memory APIs: %s", exc)
        return False
