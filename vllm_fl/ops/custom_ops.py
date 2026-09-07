# Copyright (c) 2025 BAAI. All rights reserved.

import logging
import sys
from typing import Optional, List

from vllm.model_executor.custom_op import CustomOp, PluggableLayer
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
# fused_moe is handled separately because vLLM exposes it as a factory function.
OOT_OPS = {
    "silu_and_mul": (SiluAndMulFL, "SiluAndMul"),  # noqa F405
    "gelu_and_mul": (GeluAndMulFL, "GeluAndMul"),  # noqa F405
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
    # NOTE: fused_moe is NOT registered via PluggableLayer/CustomOp.register_oot.
    # In vllm >= 0.24.0, FusedMoE is a factory function (not a class), so the
    # PluggableLayer OOT path is incompatible.  Instead, FusedMoEFL is injected
    # via monkey-patch in register_oot_ops() below.
    # "fused_moe": (FusedMoEFL, "FusedMoE"),
    # unquantized_fused_moe_method is also handled via FusedMoEFL factory —
    # no separate registration needed.
    # "unquantized_fused_moe_method": (UnquantizedFusedMoEMethodFL, "UnquantizedFusedMoEMethod"),
}

def _patch_unquantized_moe_oracle(*, prefer_flaggems_experts: bool) -> None:
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
    from vllm_fl.ops.fused_moe.fused_moe_utils import (
        select_unquantized_moe_backend_oot,
    )

    def select_backend(moe_config):
        return select_unquantized_moe_backend_oot(
            moe_config,
            prefer_flaggems_experts=prefer_flaggems_experts,
        )

    _oracle_mod.select_unquantized_moe_backend = select_backend
    # Also patch the import in unquantized_fused_moe_method module
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as _method_mod
    _method_mod.select_unquantized_moe_backend = select_backend
    logger.info(
        "Patched select_unquantized_moe_backend for OOT execution "
        "(prefer_flaggems_experts=%s)",
        prefer_flaggems_experts,
    )


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
        _patch_unquantized_moe_oracle(prefer_flaggems_experts=False)
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

    # FusedMoE is not present in OOT_OPS because it is a factory, but it still
    # follows the same whitelist/blacklist policy as registered OOT classes.
    active_whitelist = env_whitelist if env_whitelist is not None else whitelist
    fused_moe_enabled = (
        active_whitelist is None or "fused_moe" in active_whitelist
    ) and "fused_moe" not in blacklist

    # Apply blacklist to class-based OOT ops.
    ops_to_register = [op for op in ops_to_register if op not in blacklist]

    # If fused_moe is excluded (blacklisted or not in the active whitelist),
    # patch the upstream oracle so in-tree FusedMoE works on OOT platforms.
    if not fused_moe_enabled:
        _patch_unquantized_moe_oracle(prefer_flaggems_experts=False)

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

    # --- FusedMoE monkey-patch (vllm >= 0.24.0) ---
    # FusedMoE is a factory function in vllm 0.24.0+, not a PluggableLayer
    # subclass, so it cannot be registered via CustomOp/PluggableLayer.register_oot.
    # Instead we replace the factory function in the two places vllm imports it
    # from, so all model code transparently gets FusedMoEFL.
    if fused_moe_enabled:
        # Some model modules import the upstream FusedMoE factory before the
        # WorkerFL constructor runs.  Patching only the package-level factory
        # cannot rewrite those already-bound symbols.  Patch the method's
        # backend oracle as the authoritative v0.24 path as well, so both new
        # and already-imported factories select TritonExpertsFL.
        _patch_unquantized_moe_oracle(prefer_flaggems_experts=True)
        _patch_fused_moe_factory()


def _patch_fused_moe_factory() -> None:
    """Replace the FusedMoE factory function with FusedMoEFL in all relevant
    vllm modules so that model code picks up the FL version automatically."""
    import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

    if getattr(_fused_moe_layer, "FusedMoE", None) is FusedMoEFL:  # noqa F405
        # Already patched — idempotent.
        return

    native_factory = _fused_moe_layer.FusedMoE

    # Model modules such as qwen3_next may execute
    # ``from vllm...fused_moe import FusedMoE`` during registry inspection,
    # before WorkerFL installs OOT operators.  Replacing only the defining
    # modules leaves those already-bound globals pointing at the native
    # factory.  Rewrite exact identity matches in loaded vLLM model modules;
    # do not touch unrelated callables that merely share the same name.
    patched_model_modules = 0
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("vllm.model_executor.models."):
            continue
        if getattr(module, "FusedMoE", None) is native_factory:
            setattr(module, "FusedMoE", FusedMoEFL)  # noqa F405
            patched_model_modules += 1

    # Patch defining modules for imports that occur after this point.
    _fused_moe_layer.FusedMoE = FusedMoEFL  # noqa F405
    _fused_moe_pkg.FusedMoE = FusedMoEFL   # noqa F405
    logger.info(
        "Monkey-patched FusedMoE factory -> FusedMoEFL "
        "(already-bound model modules=%d)",
        patched_model_modules,
    )
