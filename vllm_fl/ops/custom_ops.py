# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe import *  # noqa F403 F401
from .deepseek_v4_attention import DeepseekV4MultiHeadLatentAttentionFLWrapper # noqa F403 F401

logger = logging.getLogger(__name__)

# Mapping from OOT operator name (op_name, internal/whitelist) to (class, registration_name).
# registration_name is passed to CustomOp.register_oot and must match what vLLM uses
# when looking up the OOT op (typically the base class name).
# item example as follows:
# op_name: (class, registration_name of vllm's CustomOp.register_oot)
# note: cannot control inner gems op of UnquantizedFusedMoEMethodFL via env variable.
OOT_OPS = {
    "silu_and_mul": (SiluAndMulFL, "SiluAndMul"),  # noqa F405
    "gelu_and_mul": (GeluAndMulFL, "GeluAndMul"),  # noqa F405
    "silu_and_mul_with_clamp": (SiluAndMulWithClampFL, "SiluAndMulWithClamp"),  # noqa F405
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
    "fused_moe": (FusedMoEFL, "FusedMoE"),  # noqa F405
    "unquantized_fused_moe_method": (
        UnquantizedFusedMoEMethodFL,  # noqa F405
        "UnquantizedFusedMoEMethod",
    ),
    "deepseek_v4_multi_head_latent_attention": (
        DeepseekV4MultiHeadLatentAttentionFLWrapper,
        "DeepseekV4MultiHeadLatentAttentionWrapper"
    ),  # noqa F405
    "gated_layer": (GateLinearFL, "GateLinear"),
}

def _patch_unquantized_moe_oracle() -> None:
    """
    Monkey-patch the upstream select_unquantized_moe_backend so it does not
    short-circuit to (OOT, None) on our platform.  Instead it falls through
    to the normal CUDA/ROCm backend priority selection — the same logic that
    select_unquantized_moe_backend_oot uses.

    This is needed when FusedMoEFL is NOT registered (PREFER_ENABLED=0 or
    fused_moe blacklisted): without the patch, the in-tree UnquantizedFusedMoEMethod
    would get (OOT, None), skip _setup_kernel, and crash at inference time.
    """
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as _oracle_mod
    from vllm_fl.ops.fused_moe.fused_moe_utils import select_unquantized_moe_backend_oot
    _oracle_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    # Also patch the import in unquantized_fused_moe_method module
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as _method_mod
    _method_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    logger.info("Patched select_unquantized_moe_backend to bypass OOT short-circuit")


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.

    Operators in VLLM_FL_OOT_BLACKLIST or platform config oot_blacklist
    will be excluded from registration.

    When fused_moe is not registered (PREFER_ENABLED=0 or blacklisted),
    the upstream select_unquantized_moe_backend oracle is monkey-patched
    so it picks native CUDA backends instead of returning (OOT, None).
    """
    from vllm_fl.utils import get_oot_blacklist, get_oot_whitelist, is_oot_enabled, use_flaggems_op

    # Check if OOT registration is enabled
    if not is_oot_enabled():
        # Patch the upstream oracle so in-tree FusedMoE works on this platform.
        _patch_unquantized_moe_oracle()
        return

    # Get blacklist (from env var or platform config)
    blacklist = get_oot_blacklist() or []

    # Determine which operators to register
    env_whitelist = get_oot_whitelist()
    if env_whitelist is not None:
        ops_to_register = env_whitelist
    elif whitelist is not None:
        ops_to_register = whitelist
    else:
        ops_to_register = list(OOT_OPS.keys())

    # Apply blacklist
    ops_to_register = [op for op in ops_to_register if op not in blacklist]

    # If fused_moe is excluded (blacklisted or not in whitelist), patch the
    # upstream oracle so the in-tree FusedMoE doesn't crash on OOT platforms.
    if "fused_moe" not in ops_to_register:
        _patch_unquantized_moe_oracle()

    for op_name in ops_to_register:
        if op_name not in OOT_OPS:
            logger.warning(f"OOT op '{op_name}' not found in OOT_OPS, skipping.")
            continue

        # unquantized_fused_moe_method only registers when use_flaggems_op is True
        if op_name == "unquantized_fused_moe_method" and not use_flaggems_op(op_name):
            logger.debug(f"Skipping '{op_name}': use_flaggems_op returned False")
            continue

        op_cls, registration_name = OOT_OPS[op_name]
        logger.info(f"Registering oot op: {op_name} as '{registration_name}'")
        if issubclass(op_cls, PluggableLayer):
            PluggableLayer.register_oot(_decorated_layer_cls=op_cls, name=registration_name)
        else:
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=registration_name)
        # Apply Ascend NPU monkey-patches if running on NPU.
        # These replace upstream module-level functions (e.g. in qwen3_next) with
        # Ascend implementations that bypass the CustomOp/dispatch path.
        from vllm.platforms import current_platform
        if current_platform.device_type == "npu":
            from vllm_fl.dispatch.backends.vendor.ascend.patch import apply_ascend_patches
            apply_ascend_patches()

        # Apply Sunrise/PTPU monkey-patches if running on PTPU.
        if current_platform.device_type == "ptpu":
            from vllm_fl.dispatch.backends.vendor.sunrise.patch import apply_sunrise_patches
            apply_sunrise_patches()
