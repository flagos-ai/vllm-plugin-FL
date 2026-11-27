# Copyright (c) 2026 BAAI. All rights reserved.

import logging

from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.causal_conv1d import apply_causal_conv1d_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.index_fill import apply_index_fill_gcu_patch
from .impl.inductor import apply_inductor_gcu_patch
from .impl.kv_zeroer import apply_kv_zeroer_gcu_patch
from .impl.mamba_copy import apply_mamba_copy_gcu_patch
from .impl.sampler import apply_sampler_cpu_detour
from .impl.reference_moe import apply_reference_moe_gcu_patch
from .impl.fused_sigmoid_gating import apply_fused_sigmoid_gating_gcu_patch
from .impl.moe_align_block_size import apply_moe_align_block_size_gcu_patch
from .impl.norm_gated_compat import install_rms_norm_gated_opaque

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_gcu_patches() -> None:
    """Apply all GCU-specific kernel / model monkey-patches."""
    global _patches_applied
    if _patches_applied:
        return

    # torch_gcu 2.11 bundles a flag_gems whose aten::exponential_ Python
    # kernel lacks the ``generator`` argument that vLLM's TopKTopP sampler
    # passes; install the generator-aware shim before any sampling runs.
    from vllm_fl.patches.exponential_compat import apply_exponential_compat

    apply_exponential_compat()
    apply_bilinear_pos_embed_gcu_patch()
    apply_causal_conv1d_gcu_patch()
    apply_chunk_delta_h_gcu_patch()
    apply_sampler_cpu_detour()
    apply_fused_recurrent_packed_decode_gcu_patch()
    apply_kv_zeroer_gcu_patch()
    apply_mamba_copy_gcu_patch()
    apply_index_fill_gcu_patch()
    apply_inductor_gcu_patch()
    # Vendor-neutral fallbacks adjusted for GCU only (int32 routing, chunked
    # per-expert counts, no gather on the sampler hot path).
    apply_reference_moe_gcu_patch()
    apply_fused_sigmoid_gating_gcu_patch()
    # gcu300 Inductor cannot register-allocate the fused gated-RMSNorm kernel;
    # keep it opaque (no-op unless running on gcu300 / env opt-out).
    install_rms_norm_gated_opaque()
    # vLLM 0.32 passes an 8th scatter_idx arg that the GCU _moe_C lacks.
    apply_moe_align_block_size_gcu_patch()
    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
