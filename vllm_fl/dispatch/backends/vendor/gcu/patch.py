# Copyright (c) 2026 BAAI. All rights reserved.

import logging

from .impl.apply_moe_activation_gcu import apply_moe_activation_gcu_patch
from .impl.fp8_config import apply_fp8_config_gcu_patch
from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.moe_align_block_size import apply_moe_align_block_size_gcu_patch
from .impl.moe_sum import apply_moe_sum_gcu_patch
from .impl.per_token_group_quant_fp8 import (
    apply_per_token_group_quant_fp8_gcu_patch,
)
from .impl.quant_fp8 import apply_quant_fp8_gcu_patch
from .impl.fused_moe import apply_fused_moe_triton_kernel_gcu_patch

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_gcu_patches() -> None:
    """Apply all GCU-specific kernel / model monkey-patches."""
    global _patches_applied
    if _patches_applied:
        return
    
    apply_bilinear_pos_embed_gcu_patch()
    apply_chunk_delta_h_gcu_patch()
    apply_fused_recurrent_packed_decode_gcu_patch()
    apply_moe_activation_gcu_patch()
    apply_moe_align_block_size_gcu_patch()
    apply_moe_sum_gcu_patch()
    apply_per_token_group_quant_fp8_gcu_patch()
    apply_fp8_config_gcu_patch()
    apply_quant_fp8_gcu_patch()
    apply_fused_moe_triton_kernel_gcu_patch()
    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
