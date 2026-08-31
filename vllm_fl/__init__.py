# Copyright (c) 2025 BAAI. All rights reserved.

import importlib
import logging
import os
import platform
import sys
from importlib import metadata

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


def _is_arm_cpu_build() -> bool:
    """Return whether vLLM is an AArch64 CPU build."""
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    target_is_cpu = os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu"
    try:
        return "cpu" in metadata.version("vllm").lower() or target_is_cpu
    except metadata.PackageNotFoundError:
        return target_is_cpu


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
    """Register fallback schemas when neither vLLM extension ABI is present."""
    for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            importlib.import_module(module_name)
            return
        except (ImportError, OSError):
            continue

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def register():
    """Register the FL platform."""
    if _is_arm_cpu_build():
        logger.info("[vllm_fl] ARM CPU -> native-backed FL CPU platform")
        return "vllm_fl.platform_cpu.CpuPlatformFL"

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
    from vllm.model_executor.models import ModelRegistry

    # Register before the short-lived registry probe exits. The subprocess
    # resolves architectures without loading the ARM runtime, but it still
    # needs to know which module owns the DFlash2 checkpoint entry point.
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_fl.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )

    # General plugins are loaded independently in spawned model-inspection and
    # worker processes, so all runtime compatibility hooks must be idempotent.
    from vllm_fl.patches.moe_sum import patch_vllm_moe_sum
    from vllm_fl.patches.qwen3_5_text import apply_qwen3_5_text_patches

    apply_qwen3_5_text_patches()

    from vllm.platforms import current_platform
    if current_platform.device_type == "cpu" and _is_arm_cpu_build():
        # Registry inspection only needs model declarations. Loading the ARM
        # runtime here would consume process locks before the engine starts.
        if os.path.basename(sys.argv[0]) == "registry.py":
            return

        # Spec-decode and hybrid-cache compatibility apply to BF16 and every
        # quant backend. FlagGems model kernels are selected independently.
        from vllm_fl.patches.arm_cpu_vllm_0240 import (
            install_arm_cpu_vllm_0240_compat,
        )

        install_arm_cpu_vllm_0240_compat()

        int8_enabled = os.environ.get("FL_CPU_INT8", "0").lower()
        if int8_enabled not in {"0", "1", "false", "true"}:
            raise ValueError("FL_CPU_INT8 must be one of: 0, 1, false, true")
        if int8_enabled in {"1", "true"}:
            backend = os.environ.get(
                "FL_CPU_INT8_BACKEND", "libtriton_jit"
            ).lower()
            if backend == "torchpack":
                from vllm_fl.ops.cpu_int8_pack import enable_int8

                enable_int8()
                return
            if backend != "libtriton_jit":
                raise ValueError(
                    "FL_CPU_INT8_BACKEND must be 'libtriton_jit' or "
                    "'torchpack'"
                )
            from vllm_fl.ops.cpu_qwen_runtime import enable_qwen_runtime

            enable_qwen_runtime()
            return

        int4_enabled = os.environ.get("FL_CPU_INT4", "0").lower()
        if int4_enabled not in {"0", "1", "false", "true"}:
            raise ValueError("FL_CPU_INT4 must be one of: 0, 1, false, true")
        if int4_enabled in {"1", "true"}:
            backend = os.environ.get(
                "FL_CPU_INT4_BACKEND", "libtriton_jit"
            ).lower()
            if backend != "libtriton_jit":
                raise ValueError("FL_CPU_INT4_BACKEND must be 'libtriton_jit'")
            from vllm_fl.ops.cpu_qwen_runtime import enable_qwen_runtime

            enable_qwen_runtime()
        else:
            logger.info("[vllm_fl] FL_CPU_INT4=0 -> bf16")
        return

    patch_vllm_moe_sum()

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
