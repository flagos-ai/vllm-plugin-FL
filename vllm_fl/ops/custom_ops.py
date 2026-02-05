# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe import *  # noqa F403 F401

logger = logging.getLogger(__name__)

# Mapping from OOT operator name (op_name, internal/whitelist) to (class, registration_name).
# registration_name is passed to CustomOp.register_oot and must match what vLLM uses
# when looking up the OOT op (typically the base class name).
# item example as follows:
# op_name: (class, registration_name of vllm's CustomOp.register_oot)
# note: cannot control inner gems op of UnquantizedFusedMoEMethodFL via env variable.
OOT_OPS = {
    "silu_and_mul": (SiluAndMulFL, "SiluAndMul"),  # noqa F405
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
    "fused_moe": (FusedMoEFL, "FusedMoE"),  # noqa F405
    "unquantized_fused_moe_method": (
        UnquantizedFusedMoEMethodFL,  # noqa F405
        "UnquantizedFusedMoEMethod",
    ),
}


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
        logger.info(f"Registering oot op: {op_name} as '{registration_name}'")
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=registration_name)
