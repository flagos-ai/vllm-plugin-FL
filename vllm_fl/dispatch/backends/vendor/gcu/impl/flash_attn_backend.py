# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GCU300 native-FLASH_ATTN backend enablement patch.

The empty vLLM build (VLLM_TARGET_DEVICE=empty) strips ``vllm._C``, so
``vllm/v1/attention/backends/fa_utils.py`` never binds the flash-attention ops
for GCU (it only binds them on cuda/xpu/rocm). Consequently
``is_flash_attn_varlen_func_available()`` returns False and
``vllm/v1/attention/backends/flash_attn.py`` skips its conditional import of
``reshape_and_cache_flash`` / ``flash_attn_varlen_func`` / ``get_scheduler_metadata``
/ ``flash_attn_supports_sinks`` (NameError at do_kv_cache_update otherwise).

On enflame/GCU those ops ARE available, just from different sources:
  * flash_attn_varlen_func, get_scheduler_metadata -> vendor flash_attn.vllm_flash_attn
  * reshape_and_cache_flash                        -> flag_gems.fused (triton kernel)
  * flash_attn_supports_sinks                      -> fa_utils itself (always defined)

This patch binds those ops onto ``fa_utils`` and forces
``is_flash_attn_varlen_func_available()`` to True, so vLLM's native FLASH_ATTN
backend works unmodified. It is robust to import ordering:
  * if flash_attn.py is not yet imported, patching fa_utils is enough (its
    conditional import at load time will pick up the names + the True gate);
  * if flash_attn.py was already imported (its gate ran and skipped the import),
    we also inject the four names + the gate into flash_attn.py's namespace.

Applied only when the GCU backend loads (via apply_gcu_patches), so no other
vendor is affected and vLLM site-packages is left pristine.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_FA_UTILS = "vllm.v1.attention.backends.fa_utils"
_FLASH_ATTN = "vllm.v1.attention.backends.flash_attn"


def apply_flash_attn_backend_gcu_patch() -> None:
    if getattr(sys.modules.get(_FA_UTILS), "_gcu_flash_attn_patched", False):
        return

    try:
        from flash_attn.vllm_flash_attn import (
            flash_attn_varlen_func,
            get_scheduler_metadata,
        )
        from flag_gems.fused import reshape_and_cache_flash
    except ImportError as e:
        # Best-effort: vendor flash_attn or flag_gems missing -> leave vLLM as-is.
        logger.warning(
            "GCU: flash_attn backend patch skipped (missing dep: %s)", e
        )
        return

    import importlib

    fa_utils = importlib.import_module(_FA_UTILS)

    # Bind the ops onto fa_utils so flash_attn.py's conditional import (if it
    # runs after this patch) resolves them, and flip the availability gate.
    fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
    fa_utils.get_scheduler_metadata = get_scheduler_metadata
    fa_utils.reshape_and_cache_flash = reshape_and_cache_flash
    fa_utils._GCU_FLASH_ATTN_AVAILABLE = True
    fa_utils._gcu_flash_attn_patched = True
    fa_utils.is_flash_attn_varlen_func_available = lambda: True

    # If flash_attn.py already imported (its gate ran False and it skipped the
    # conditional import), inject the four names + the gate directly.
    flash_attn_mod = sys.modules.get(_FLASH_ATTN)
    if flash_attn_mod is not None:
        flash_attn_mod.flash_attn_varlen_func = flash_attn_varlen_func
        flash_attn_mod.get_scheduler_metadata = get_scheduler_metadata
        flash_attn_mod.reshape_and_cache_flash = reshape_and_cache_flash
        flash_attn_mod.flash_attn_supports_sinks = fa_utils.flash_attn_supports_sinks
        flash_attn_mod.is_flash_attn_varlen_func_available = lambda: True

    logger.info(
        "GCU: enabled native FLASH_ATTN backend "
        "(vendor flash_attn_varlen_func + flag_gems reshape_and_cache_flash)"
    )
