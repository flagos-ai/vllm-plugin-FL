# Copyright (c) 2026 BAAI. All rights reserved.

"""Import compatibility for NVIDIA with upstream's kernel-free wheel.

The CUDA platform imports the stable extension unconditionally in vLLM 0.32.
An empty wheel has no such module. The placeholder below only allows platform
discovery; it neither supplies kernels nor advertises their availability.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import types

_NATIVE_MODULES = ("vllm._C", "vllm._C_stable_libtorch")
_EMPTY_MARKER = "_vllm_fl_empty_placeholder"


def has_native_vllm_kernels() -> bool:
    """Inspect installed extensions without mistaking our placeholder for one."""
    for name in _NATIVE_MODULES:
        module = sys.modules.get(name)
        if module is not None and getattr(module, _EMPTY_MARKER, False):
            continue
        spec = importlib.util.find_spec(name)
        if (
            spec is not None
            and spec.origin
            and any(
                spec.origin.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
        ):
            return True
    return False


def prepare_nvidia_empty_imports() -> None:
    """Allow CudaPlatform discovery while preserving errors in native wheels."""
    if has_native_vllm_kernels():
        # An installed extension with a broken ABI is not an empty build.
        # Propagate its load failure instead of hiding it behind the shim.
        for name in _NATIVE_MODULES:
            if importlib.util.find_spec(name) is not None:
                importlib.import_module(name)
        return

    name = "vllm._C_stable_libtorch"
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    setattr(module, _EMPTY_MARKER, True)
    sys.modules[name] = module
