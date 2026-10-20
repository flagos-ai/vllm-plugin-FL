# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise import-time patches (0.24 full port — 2026-09-20).

All-patches port: the 0.2.1 master ``__init__`` auto-ran seven import-time
``apply_patch()`` hooks. This module now mirrors that, guarding each so a
0.24 API mismatch degrades to a skipped patch instead of an import crash.

* **Import-time** (this module): FlagGems pointwise fast-path, FLA/GDN kernel
  rebinds, GDN core-attn buffer reuse, compressed-tensors INT8 enablement,
  fused-MoE tile config, optional decode profiler, torch profiler bridge.
* **Deferred** (``patch.apply_sunrise_patches``): stream shims, FlagCX
  comm/collectives, cudagraph, sampler/penalties, distributed runtime, OOT
  layer registration, ``_moe_C`` dispatch shims, MoE/int8 routing.
"""

import logging

logger = logging.getLogger(__name__)

from . import patch_fla_ops  # noqa: F401,E402
from . import patch_gdn_core_attn_buf  # noqa: F401,E402

# FlagGems pointwise fast-path (must run before any FlagGems op dispatch).
try:
    from . import patch_pointwise

    patch_pointwise.apply_patch()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_pointwise.apply_patch failed")

# Route vLLM FLA/GDN kernels to PTPU sgl_kernel (Qwen3.5/3.6 linear-attention).
try:
    patch_fla_ops.apply_patch()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_fla_ops.apply_patch failed")

# Reuse GDN core_attn_out buffer (eliminates per-iter zeros_kernel).
try:
    patch_gdn_core_attn_buf.apply_patch()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_gdn_core_attn_buf.apply_patch failed")

# compressed-tensors INT8 (W8A8) enablement on PTPU; no-op for BF16 models.
try:
    from . import patch_int8_native

    patch_int8_native.enable_native_int8()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_int8_native.enable_native_int8 failed")

# Fused-MoE tile config from FlagGems' sunrise backend (not vLLM NVIDIA heuristic).
try:
    from . import patch_moe_config

    patch_moe_config.apply_patch()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_moe_config.apply_patch failed")

# Optional per-operator decode-step profiler (env-gated).
try:
    from . import patch_profile_decode

    patch_profile_decode.install()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_profile_decode.install failed")

# torch.profiler PrivateUse1 bridge on PTPU.
try:
    from . import patch_profiler

    patch_profiler.install()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_profiler.install failed")

# Override MLA prefill backend selection on PTPU (0.24 MLAPrefillBackend API).
try:
    from . import patch_mla_prefill

    patch_mla_prefill.apply_patch()
except Exception:  # pragma: no cover - defensive
    logger.exception("import-time patch_mla_prefill.apply_patch failed")

__all__ = [
    "patch_fla_ops",
    "patch_gdn_core_attn_buf",
    "patch_pointwise",
    "patch_int8_native",
    "patch_moe_config",
    "patch_profile_decode",
    "patch_profiler",
    "patch_mla_prefill",
]
