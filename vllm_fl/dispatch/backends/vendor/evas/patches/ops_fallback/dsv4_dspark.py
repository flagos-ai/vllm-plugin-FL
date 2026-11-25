# Copyright (c) 2026 BAAI. All rights reserved.
"""DSV4 DSpark drafter forward on LPU — rebind the two tilelang MHC kernels.

From vLLM 0.25 the drafter is ``nvidia/dspark``; its ``forward`` binds two
tilelang MHC kernels at module scope and calls them directly, so the
``CustomOp`` fallbacks in ``layers/mhc.py`` never get a say. The patch rebinds
the dspark module's own copies to the stand-ins ``dsv4_mtp`` uses
(``flag_gems.mhc_post`` / ``model_compat.lpu_hc_head_fused_kernel``).
Idempotent; a no-op off LPU and on cores that predate the module (0.24).

"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False

_DSPARK_MODULE = "vllm.models.deepseek_v4.nvidia.dspark"


def apply_lpu_dsv4_dspark_patch() -> bool | None:
    """Rebind the DSpark drafter's two module-scope tilelang kernels on LPU.

    Inert unless the DSpark drafter is instantiated, which happens only when
    speculative decoding selects the dspark method.

    Returns False when the drafter module is missing on a core that is
    supposed to ship it (the spec's ``min_vllm`` floor keeps the registry from
    applying this patch on older cores), so the miss reaches the apply log.
    """
    global _PATCHED

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return None
    if _PATCHED:
        return None

    import importlib.util

    if importlib.util.find_spec(_DSPARK_MODULE) is None:
        logger.warning(
            "dsv4_dspark patch did not install: %s is missing (upstream "
            "rename or a core without the dspark drafter?)",
            _DSPARK_MODULE,
        )
        return False

    import flag_gems
    import vllm.models.deepseek_v4.nvidia.dspark as _dspark

    from ..model_compat import lpu_hc_head_fused_kernel

    _dspark.mhc_post_tilelang = flag_gems.mhc_post
    _dspark.hc_head_fused_kernel_tilelang = lpu_hc_head_fused_kernel

    _PATCHED = True
