# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_vllm_matcher_utils():
    """Pre-patch vllm.compilation.matcher_utils to avoid AttributeError on MUSA.

    On MUSA/MetaX platforms, vllm's _C extension cannot be loaded because it
    links against libcudart.so.12 which is not present. This causes
    vllm/compilation/matcher_utils.py to crash at import time (line 26) when
    it tries to access torch.ops._C.rms_norm / fused_add_rms_norm /
    rotary_embedding.

    We patch the module by replacing it with a safe stub *before* the
    EngineCore subprocess imports it, so the rest of vllm can continue to
    import cleanly. The actual OOT ops will be registered later by
    register_oot_ops() in worker.py.
    """
    import sys
    import types
    import torch

    # Only patch when the _C ops are genuinely missing (i.e. MUSA platform).
    _c_ns = getattr(torch.ops, "_C", None)
    if _c_ns is not None and hasattr(_c_ns, "rms_norm"):
        return  # CUDA path: ops already loaded, nothing to do.

    # If matcher_utils is already imported and succeeded, nothing to do.
    if "vllm.compilation.matcher_utils" in sys.modules:
        return

    # Build a minimal stub module that exposes the names matcher_utils needs.
    stub = types.ModuleType("vllm.compilation.matcher_utils")
    stub.RMS_OP = None
    stub.RMS_ADD_OP = None
    stub.ROTARY_OP = None

    # Provide a no-op get_matching_ops so call-sites don't crash.
    def get_matching_ops(*args, **kwargs):
        return []

    stub.get_matching_ops = get_matching_ops
    sys.modules["vllm.compilation.matcher_utils"] = stub
    logger.info(
        "vllm_fl: patched vllm.compilation.matcher_utils with stubs "
        "(MUSA platform — libcudart.so.12 not available)."
    )


def register():
    """Register the FL platform."""
    _patch_vllm_matcher_utils()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
