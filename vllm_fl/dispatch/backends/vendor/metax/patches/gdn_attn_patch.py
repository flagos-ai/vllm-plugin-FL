# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""
Patch GatedDeltaNetAttention to be compatible with torch._dynamo (torch.compile).

torch._dynamo in MetaX's torch 2.8.0 does not correctly handle `hasattr`
guards on dynamically-created attributes. When `create_in_proj_qkvz=True`
(the non-LoRA path), `in_proj_qkv` is never created, but Dynamo's symbolic
tracing incorrectly enters the `hasattr(self, "in_proj_qkv")` branch.

Fix:
1. Patch __init__ to always define `self.in_proj_qkv = None` when using merged path
2. Monkey-patch `hasattr` in forward_cuda/forward_xpu's global namespace to check
   `is not None` instead of attribute existence
"""

import logging

logger = logging.getLogger(__name__)

try:
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
    )
    import functools

    # --- Step 1: Patch __init__ to always define in_proj_qkv ---
    _orig_init = GatedDeltaNetAttention.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Ensure in_proj_qkv always exists as an attribute.
        # When create_in_proj_qkvz=True (non-LoRA), in_proj_qkv won't be created.
        # Set it to None so our custom hasattr can check `is not None`.
        if not hasattr(self, "in_proj_qkv"):
            # Use object.__setattr__ to bypass nn.Module's parameter registration
            object.__setattr__(self, "in_proj_qkv", None)

    GatedDeltaNetAttention.__init__ = _patched_init

    # --- Step 2: Monkey-patch hasattr in forward_cuda/forward_xpu globals ---
    _orig_forward_cuda = GatedDeltaNetAttention.forward_cuda
    _cuda_globals = _orig_forward_cuda.__globals__

    # Save original hasattr
    _orig_hasattr = _cuda_globals.get("hasattr", hasattr)

    def _custom_hasattr(obj, name):
        """
        Custom hasattr that checks `is not None` for GatedDeltaNetAttention.in_proj_qkv.
        This makes Dynamo's attribute guard work correctly.
        """
        if name == "in_proj_qkv" and isinstance(obj, GatedDeltaNetAttention):
            # Return False if it's None (even though attribute exists)
            return getattr(obj, "in_proj_qkv", None) is not None
        return _orig_hasattr(obj, name)

    # Replace hasattr in forward_cuda's globals
    _cuda_globals["hasattr"] = _custom_hasattr

    # Do the same for forward_xpu if it exists
    if hasattr(GatedDeltaNetAttention, "forward_xpu"):
        _xpu_globals = GatedDeltaNetAttention.forward_xpu.__globals__
        _xpu_globals["hasattr"] = _custom_hasattr

    logger.info("Patched GatedDeltaNetAttention for torch.compile compatibility")

except ImportError:
    logger.debug("GatedDeltaNetAttention not found, skipping patch")
except Exception as e:
    logger.warning(f"Failed to patch GatedDeltaNetAttention: {e}")
