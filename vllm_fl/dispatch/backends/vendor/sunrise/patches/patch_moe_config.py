# Copyright (c) 2026 BAAI. All rights reserved.

"""Take the fused-MoE tile config from FlagGems' sunrise backend.

FlagGems keeps a PTPU-tuned MoE heuristic in
``flag_gems/runtime/backend/_sunrise/fused/fused_moe.py``
(``_sunrise_get_default_config``), but it is installed by a context manager that
only wraps ``flag_gems.fused_experts_impl``. The plugin never reaches that
wrapper on PTPU: the monolithic fast path in ``vllm_fl/ops/fused_moe/
fused_moe_utils.py`` is gated on ``current_platform.is_cuda()``, so we go through
``TritonExpertsFL``, which picks tiles with vLLM's ``try_get_optimal_moe_config``
and hands the result to FlagGems' kernel. FlagGems launches whatever it is given.

vLLM's heuristic is tuned for NVIDIA and derives the tile from the token count
alone. On PTPU its prefill choice (``BLOCK_SIZE_M=128``, ``BLOCK_SIZE_N=128``,
8 warps) puts the 128x128 f32 accumulator over the register budget, which wedged
the w2 grouped GEMM under sustained load.

So rebind ``try_get_optimal_moe_config`` to a wrapper that asks FlagGems' sunrise
backend instead. Config *ownership* moves to FlagGems, which is where the people
tuning these kernels work; this module only routes.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_PATCHED = False
# gemm1 (w13) and gemm2 (w2) deliberately share one config: BLOCK_SIZE_M also
# sizes the moe_align_block_size padding that both GEMMs index into, so they
# cannot disagree on it. Picking once matches what the callers already did.
_GEMM_STAGE = "gemm1"


def _sunrise_moe_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
    """FlagGems-sourced replacement for vLLM's ``try_get_optimal_moe_config``."""
    from flag_gems.fused import fused_moe as gems_moe
    from flag_gems.runtime.backend._sunrise.fused.fused_moe import (
        _sunrise_moe_config_patch,
    )

    # Keep VLLM_FL_MOE_FORCE_CONFIG (see patch.py) authoritative for debugging.
    from vllm.model_executor.layers.fused_moe import get_config

    override = get_config()
    if override:
        return override

    # FlagGems' picker takes the expert count explicitly and distinguishes the
    # two GEMM stages, neither of which vLLM's signature carries.
    with _sunrise_moe_config_patch():
        return gems_moe.try_get_optimal_moe_config(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            w1_shape[0],
            block_shape=block_shape,
            gemm_stage=_GEMM_STAGE,
        )


def apply_patch() -> bool:
    """Rebind ``try_get_optimal_moe_config`` to the FlagGems sunrise picker."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
    except Exception as exc:  # noqa: BLE001
        logger.debug("moe-config: vLLM fused_moe module unavailable (%s)", exc)
        return False

    original = getattr(vllm_moe, "try_get_optimal_moe_config", None)
    if original is None:
        logger.debug("moe-config: try_get_optimal_moe_config not found; skip.")
        return False
    if getattr(original, "_fl_sunrise_moe_config", False):
        _PATCHED = True
        return True

    def _try_get_optimal_moe_config(
        w1_shape, w2_shape, top_k, dtype, M, block_shape=None
    ):
        try:
            return _sunrise_moe_config(
                w1_shape, w2_shape, top_k, dtype, M, block_shape=block_shape
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "moe-config: FlagGems sunrise MoE config unavailable (%s); "
                "falling back to vLLM's NVIDIA-tuned heuristic, which has been "
                "observed to hang the w2 GEMM on PTPU.",
                exc,
            )
            return original(w1_shape, w2_shape, top_k, dtype, M, block_shape)

    _try_get_optimal_moe_config._fl_sunrise_moe_config = True  # type: ignore[attr-defined]
    _try_get_optimal_moe_config._fl_original = original  # type: ignore[attr-defined]

    # Patch the defining module, then rebind the plugin modules that already did
    # ``from ...fused_moe import try_get_optimal_moe_config`` (they hold the
    # function by value): vllm_fl/ops/fused_moe/{fused_moe,fused_moe_utils}.py.
    vllm_moe.try_get_optimal_moe_config = _try_get_optimal_moe_config
    rebound = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "try_get_optimal_moe_config", None) is original:
            module.try_get_optimal_moe_config = _try_get_optimal_moe_config
            rebound += 1

    _PATCHED = True
    logger.info(
        "moe-config: fused-MoE tile config now comes from FlagGems' sunrise "
        "backend (_sunrise_get_default_config); rebound %d importer(s).",
        rebound,
    )
    return True
