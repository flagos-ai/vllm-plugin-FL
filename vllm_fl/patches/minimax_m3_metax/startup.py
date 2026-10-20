# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Ordered opt-in startup through installed FL and operator packages."""

import importlib.abc
import os
import sys

from vllm_fl.minimax_m3_config import load, thaw

_LIBRARIES_CONFIGURED = False


class VendorFrameworkGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "vllm_metax",
            "m3_metax",
            "m3_runtime",
            "m3_config",
            "m3_kernels",
        }:
            raise ImportError("Forbidden legacy/framework dependency: " + fullname)
        return None


def install_import_guard():
    if any(n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules):
        raise RuntimeError("vllm_metax was already imported")
    if not any(isinstance(x, VendorFrameworkGuard) for x in sys.meta_path):
        sys.meta_path.insert(0, VendorFrameworkGuard())


def configure_libraries():
    global _LIBRARIES_CONFIGURED
    if _LIBRARIES_CONFIGURED:
        return
    from vllm_fl.ops.minimax_m3.linear import configure as configure_linear

    profile = load()
    configure_linear(thaw(profile.linear))
    _LIBRARIES_CONFIGURED = True


def register_platform():
    from .compat import ensure_dtype_alias, install_platform_hooks
    from .platform_compat import _patch_torch_accelerator
    from vllm_fl import _patch_custom_ops, _patch_transformers_compat

    load()
    install_import_guard()
    ensure_dtype_alias()
    install_platform_hooks(
        _patch_torch_accelerator, _patch_custom_ops, _patch_transformers_compat
    )
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return "vllm_fl.patches.minimax_m3_platform.MiniMaxPlatformFL"


def register_models():
    from vllm.model_executor.models.registry import ModelRegistry

    from vllm_fl import register_quant_linear

    install_import_guard()
    configure_libraries()
    register_quant_linear()
    for name in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "MiniMaxM3MTP",
    ):
        ModelRegistry.register_model(name, "vllm_fl.models.minimax_m3:" + name)
