# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe.layer import FusedMoEFL, SharedFusedMoEFL, UnquantizedFusedMoEMethodFL  # noqa F401

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
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
}

def _patch_unquantized_fused_moe_for_oot():
    """
    Fix vLLM bug: when platform is OOT, select_unquantized_moe_backend returns
    experts_cls=None and moe_kernel stays None. This breaks:
    1. is_monolithic property (calls None.is_monolithic())
    2. maybe_init_modular_kernel (tries to create modular kernel, raises)
    3. forward_oot (falls back to forward_native which asserts moe_kernel)

    We patch supports_internal_mk to return True for OOT (skips modular init),
    and forward_oot to use the plugin's fused_experts dispatch.
    """
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )

    _orig_supports_internal_mk = UnquantizedFusedMoEMethod.supports_internal_mk

    @property
    def _patched_supports_internal_mk(self) -> bool:
        if self.unquantized_backend == UnquantizedMoeBackend.OOT:
            return True
        return _orig_supports_internal_mk.fget(self)

    UnquantizedFusedMoEMethod.supports_internal_mk = _patched_supports_internal_mk

    @property
    def _patched_is_monolithic(self) -> bool:
        if self.unquantized_backend == UnquantizedMoeBackend.OOT:
            return False
        if self.unquantized_backend == UnquantizedMoeBackend.CPU:
            return True
        if self.moe_kernel is not None:
            return self.moe_kernel.is_monolithic
        if hasattr(self, "experts_cls") and self.experts_cls is not None:
            return self.experts_cls.is_monolithic()
        return False

    UnquantizedFusedMoEMethod.is_monolithic = _patched_is_monolithic

    def _patched_forward_oot(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        from vllm_fl.ops.fused_moe.fused_moe import fused_experts
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

    UnquantizedFusedMoEMethod.forward_oot = _patched_forward_oot


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.

    Operators in VLLM_FL_OOT_BLACKLIST or platform config oot_blacklist
    will be excluded from registration.
    """
    from vllm_fl.utils import get_oot_blacklist, get_oot_whitelist, is_oot_enabled, use_flaggems_op

    # Patch UnquantizedFusedMoEMethod for OOT platforms before any MoE layers are created
    _patch_unquantized_fused_moe_for_oot()

    # Check if OOT registration is enabled
    if not is_oot_enabled():
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

    for op_name in ops_to_register:
        if op_name not in OOT_OPS:
            logger.warning(f"OOT op '{op_name}' not found in OOT_OPS, skipping.")
            continue

        # unquantized_fused_moe_method only registers when use_flaggems_op is True
        if op_name == "unquantized_fused_moe_method" and not use_flaggems_op(op_name):
            logger.debug(f"Skipping '{op_name}': use_flaggems_op returned False")
            continue

        op_cls, registration_name = OOT_OPS[op_name]
        if op_cls is None:
            logger.debug(f"Skipping '{op_name}': class not available in this vLLM version")
            continue
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
