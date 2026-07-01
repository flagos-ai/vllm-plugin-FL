# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging
import sys

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

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


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register torch.ops._C op schemas when vllm._C is unavailable."""
    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def _patch_moe_backend_selection():
    """Patch select_unquantized_moe_backend to fall through to CUDA backends
    when no OOT FusedMoE pluggable layer is registered.

    Upstream returns (OOT, None) for any is_out_of_tree() platform.  When
    FusedMoEFL is registered, the OOT path delegates correctly.  When it is
    not (VLLM_FL_PREFER_ENABLED=0 or whitelist filtering), experts_cls=None
    triggers AttributeError.  This patch skips the is_out_of_tree() early
    return in that case so native CUDA backends (flashinfer/triton) are used.

    Implementation note: importing oracle.unquantized triggers the parent
    fused_moe/__init__.py which imports unquantized_fused_moe_method, binding
    the original function via `from ... import`.  We must patch BOTH module
    namespaces AFTER the import chain is fully resolved.

    NOTE: This function must NOT be called from register() (platform plugin)
    because it triggers a circular import (vllm.config not yet ready).
    It must be called from register_model() (general plugin) instead.
    """
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as oracle_mod
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.platforms import current_platform

    _orig = oracle_mod.select_unquantized_moe_backend

    def _patched(moe_config):
        # When FusedMoE is registered OOT, let upstream handle it normally
        if not current_platform.is_out_of_tree() or "FusedMoE" in op_registry_oot:
            return _orig(moe_config)
        # No OOT FusedMoE registered — select CUDA backends directly,
        # reusing upstream's _get_priority_backends logic
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            _get_priority_backends, backend_to_kernel_cls)
        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend,)

        if moe_config.is_lora_enabled:
            return UnquantizedMoeBackend.TRITON, backend_to_kernel_cls(
                UnquantizedMoeBackend.TRITON)

        AVAILABLE_BACKENDS = _get_priority_backends(moe_config)

        activation_format = (
            mk.FusedMoEActivationFormat.BatchedExperts
            if moe_config.moe_parallel_config.use_batched_activation_format
            else mk.FusedMoEActivationFormat.Standard
        )

        for backend in AVAILABLE_BACKENDS:
            k_cls = backend_to_kernel_cls(backend)
            supported, reason = k_cls.is_supported_config(
                k_cls, moe_config, None, None, activation_format)
            if supported:
                logger.info(
                    f"Using {backend.value} Unquantized MoE backend "
                    f"(FL fallback, no OOT FusedMoE registered).")
                return backend, k_cls

        raise NotImplementedError(
            "No Unquantized MoE backend supports the deployment configuration.")

    # Patch the oracle module (source of truth)
    oracle_mod.select_unquantized_moe_backend = _patched

    # Also patch the consumer module which already bound the original via
    # `from ... import` during the import chain triggered above.
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as uqm
    uqm.select_unquantized_moe_backend = _patched


def register():
    """Register the FL platform."""
    _patch_custom_ops()
    _patch_flash_attn_import()
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
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    _patch_moe_backend_selection()
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
