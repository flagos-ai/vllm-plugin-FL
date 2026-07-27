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
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing.

    When the bundled vllm C extension is unavailable (e.g. on ROCm/Hygon), we
    try to bridge to the standalone ``flash_attn`` package so that downstream
    code (e.g. TritonMLA) that checks ``vllm.vllm_flash_attn`` still finds a
    working ``flash_attn_varlen_func``.
    """
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")

        # Try bridging to the standalone flash_attn package.
        _fa_varlen_func = None
        try:
            from flash_attn import flash_attn_varlen_func as _fa_varlen_func
        except (ImportError, ModuleNotFoundError):
            pass

        if _fa_varlen_func is not None:
            _real_fa = _fa_varlen_func

            def _wrapped_flash_attn_varlen_func(*args, **kwargs):
                """Adapter: translate vLLM's vllm_flash_attn API to standalone flash_attn."""
                # vLLM uses return_softmax_lse; standalone uses return_attn_probs
                return_lse = kwargs.pop("return_softmax_lse", False)
                # Strip kwargs the standalone flash_attn doesn't understand
                kwargs.pop("fa_version", None)
                kwargs.pop("scheduler_metadata", None)
                kwargs.pop("q_descale", None)
                kwargs.pop("k_descale", None)
                kwargs.pop("v_descale", None)
                kwargs.pop("num_splits", None)
                kwargs.pop("s_aux", None)

                # vLLM uses 'seqused_k'; standalone flash_attn 2.x doesn't have it,
                # but some ROCm builds accept it. Try passing it; strip if it fails.
                seqused_k = kwargs.pop("seqused_k", None)

                if return_lse:
                    kwargs["return_attn_probs"] = True

                try:
                    result = _real_fa(*args, **kwargs)
                except TypeError:
                    # Fallback: strip any remaining unexpected kwargs
                    kwargs.pop("return_attn_probs", None)
                    result = _real_fa(*args, **kwargs)

                if return_lse:
                    if isinstance(result, tuple) and len(result) >= 2:
                        # flash_attn returns (out, softmax_lse, S_dmask)
                        return result[0], result[1]
                    return result, None
                return result

            stub.FA2_AVAILABLE = True
            stub.FA3_AVAILABLE = False
            stub.fa_version_unsupported_reason = lambda *a, **kw: None
            stub.flash_attn_varlen_func = _wrapped_flash_attn_varlen_func
            stub.get_scheduler_metadata = None
            stub.is_fa_version_supported = lambda version, *a, **kw: version == 2
        else:
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
    from vllm import ModelRegistry

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

    # Register DeepseekV4 model
    try:
        ModelRegistry.register_model(
            "DeepseekV4ForCausalLM",
            "vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register DeepseekV4 model error: {str(e)}")


    # Register DeepseekV4 model
    try:
        ModelRegistry.register_model(
            "DeepSeekV4MTPModel",
            "vllm_fl.models.deepseek_v4_mtp:DeepSeekV4MTP"
        )
    except Exception as e:
        logger.error(f"Register DeepseekV4 model error: {str(e)}")
    # Register TeleChat mHC patches for DeepSeekV3-based models
    try:
        from vllm_fl.patches.deepseek_v2_mhc import apply_model_patches as telechat_mhc
        telechat_mhc()
    except Exception as e:
        logger.error("Register TeleChat mHC patch error: %s", str(e))
