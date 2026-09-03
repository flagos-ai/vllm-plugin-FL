# Copyright (c) 2025 BAAI. All rights reserved.
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

import os
import logging


def _bootstrap_cann_custom_op_env():
    """Make the packaged CANN custom-op package discoverable at runtime.

    The AscendC custom ops under ``vllm_fl/_cann_ops_custom`` require
    ``ASCEND_CUSTOM_OPP_PATH`` to be set *before* the CANN runtime is
    initialized (the op registration/infershape scan happens at aclInit
    time), and ``libcust_opapi.so`` to be resolvable by bare-name
    ``dlopen`` (which only consults the startup-time ``LD_LIBRARY_PATH``).
    Running this at vllm_fl package import time — before torch_npu/CANN
    initialization — makes the ops work without sourcing ``set_env.bash``.
    The libcust preload uses RTLD_LOCAL: RTLD_GLOBAL causes a double-free
    at process teardown.
    """
    vendor_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "_cann_ops_custom",
        "vendors",
        "custom_transformer",
    )
    if not os.path.isdir(vendor_dir):
        return
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor_dir not in opp_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor_dir + (
            ":" + opp_path if opp_path else ""
        )
    lib_dir = os.path.join(vendor_dir, "op_api", "lib")
    if lib_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH")
            else ""
        )
        try:
            import ctypes

            ctypes.CDLL(
                os.path.join(lib_dir, "libcust_opapi.so"), mode=ctypes.RTLD_LOCAL
            )
        except OSError:
            pass


_bootstrap_cann_custom_op_env()

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


def register():
    """Register the FL platform."""
    _patch_transformers_compat()

    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5 import Qwen3_5Config
        _CONFIG_REGISTRY["qwen3_5"] = Qwen3_5Config
    except Exception as e:
        logger.error(f"Register Qwen3.5 config in platform plugin error: {str(e)}")

    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config in platform plugin error: {str(e)}")

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry

    # Kunlunxin: patch torch.xpu.get_device_name BEFORE any import that
    # transitively loads vllm.model_executor.layers.fla.ops.utils, which
    # crashes on "Torch not compiled with XPU enabled" (see patch_fla_utils.py)
    from vllm_fl.dispatch.config.utils import get_platform_name
    if get_platform_name() == "kunlunxin":
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            ensure_fla_compat,
        )
        ensure_fla_compat()

    # Register Qwen3.5 MoE config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config error: {str(e)}")

    # Register Qwen3.5 (non-MoE) config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5 import Qwen3_5Config
        _CONFIG_REGISTRY["qwen3_5"] = Qwen3_5Config
    except Exception as e:
        logger.error(f"Register Qwen3.5 config error: {str(e)}")

    # Register Qwen3Next model
    try:
        import vllm.model_executor.models.qwen3_next as qwen3_next_module
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        qwen3_next_module.Qwen3NextForCausalLM = Qwen3NextForCausalLM
        logger.warning(
            "Qwen3NextForCausalLM has been patched to use vllm_fl.models.qwen3_next, "
            "original vLLM implementation is overridden"
        )

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM",
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register Qwen3.5 MoE model
    try:
        ModelRegistry.register_model(
            "Qwen3_5MoeForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE model error: {str(e)}")

    # Register Qwen3.5 (non-MoE) model
    try:
        ModelRegistry.register_model(
            "Qwen3_5ForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5ForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 model error: {str(e)}")

    # Register MiniCPMO model
    try:
        ModelRegistry.register_model(
            "MiniCPMO",
            "vllm_fl.models.minicpmo:MiniCPMO"
        )
    except Exception as e:
        logger.error(f"Register MiniCPMO model error: {str(e)}")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration",
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

    # Register GLM-5 (GlmMoeDsa) model
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        glm5_model()

        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

    # Register BGE-M3 pooling backport for vLLM 0.13.x
    try:
        ModelRegistry.register_model(
            "BgeM3EmbeddingModel",
            "vllm_fl.models.bge_m3:BgeM3EmbeddingModel",
        )
    except Exception as e:
        logger.error(f"Register BgeM3EmbeddingModel error: {str(e)}")

    # Ensure Qwen3NextMTP draft configs are treated as text-only even when
    # ModelConfig is reconstructed in worker processes before the platform
    # post-init patch is active.
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_mtp import (
            patch_qwen3_next_mtp_multimodal_flag,
        )
        patch_qwen3_next_mtp_multimodal_flag()
    except Exception as e:
        logger.error(f"Patch Qwen3NextMTP multimodal flag error: {str(e)}")

    # Use Ascend-compatible rejection-sampler helpers for speculative decoding.
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_mtp import (
            patch_ascend_rejection_sampler,
        )
        patch_ascend_rejection_sampler()
    except Exception as e:
        logger.error(f"Patch Ascend rejection sampler error: {str(e)}")

    # Honor VLLM_CUDAGRAPH_* environment variables so that scripts can limit
    # NPU graph capture sizes without passing --compilation-config.
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3_mtp import (
            patch_cudagraph_env_vars,
        )
        patch_cudagraph_env_vars()
    except Exception as e:
        logger.error(f"Apply cudagraph env patch error: {str(e)}")

    return "vllm_fl.platform.PlatformFL"
