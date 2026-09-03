# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise import-time patches.

Patches in this package are applied when the sunrise vendor backend is loaded
(``from vllm_fl.dispatch.backends.vendor import sunrise``). They must run
before model construction and before deferred patches in ``patch.py``.

Classification
--------------
* **Import-time** (this module): FLA/GDN kernel rebinds, FlagGems pointwise
  fast-path, GDN core-attn buffer reuse, compressed-tensors INT8 enablement,
  optional decode profiler, torch profiler PrivateUse1 bridge.
* **Deferred** (``patch.apply_sunrise_patches``): CUDA stream shims, FlagCX
  comm/collectives, cudagraph, sampler/penalties, distributed runtime, OOT
  layer registration, the ``_moe_C`` dispatch shims, and other worker-init
  hooks. INT8 also re-asserts its MoE routing there, because
  ``register_oot_ops`` installs the generic FL W8A8 selector in between.
"""

from . import patch_fla_ops
from . import patch_gdn_core_attn_buf
from . import patch_int8_native
from . import patch_moe_config
from . import patch_pointwise
from . import patch_profile_decode
from . import patch_profiler

# FlagGems pointwise fast-path (must run before any FlagGems op dispatch).
patch_pointwise.apply_patch()

# Route vLLM FLA/GDN kernels to PTPU sgl_kernel (Qwen3.5 linear-attention).
patch_fla_ops.apply_patch()

# Reuse GDN core_attn_out buffer (eliminates per-iter zeros_kernel).
patch_gdn_core_attn_buf.apply_patch()

# compressed-tensors INT8 (W8A8) enablement on PTPU; no-op for BF16 models.
patch_int8_native.enable_native_int8()

# Fused-MoE tile config from FlagGems' sunrise backend, not vLLM's NVIDIA
# heuristic (whose prefill tile overflows PTPU registers and hangs the w2 GEMM).
patch_moe_config.apply_patch()

# Optional per-operator decode-step profiler (env-gated).
patch_profile_decode.install()

# Always-on torch.profiler PrivateUse1 bridge on PTPU.
patch_profiler.install()
