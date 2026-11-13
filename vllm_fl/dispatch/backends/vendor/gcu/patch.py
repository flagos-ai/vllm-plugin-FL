# Copyright (c) 2026 BAAI. All rights reserved.

import logging

import torch

from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.causal_conv1d import apply_causal_conv1d_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fla_autotune import apply_fla_autotune_gcu_patch
from .impl.fla_grid import apply_fla_grid_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.index_fill import apply_index_fill_gcu_patch
from .impl.inductor import apply_inductor_gcu_patch
from .impl.kv_zeroer import apply_kv_zeroer_gcu_patch
from .impl.logprob import apply_logprob_gcu_patch
from .impl.mamba_copy import apply_mamba_copy_gcu_patch
from .impl.penalties import apply_penalties_gcu_patch
from .impl.sampler import apply_sampler_cpu_detour
from .impl.vocab_mask import apply_vocab_mask_gcu_patch

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_gcu_patches() -> None:
    """Apply all GCU-specific kernel / model monkey-patches."""
    global _patches_applied
    # Configuration inspection can import this package on other backends.
    # A simulated platform name does not provide the TorchGCU runtime.
    gcu = getattr(torch, "gcu", None)
    if _patches_applied or gcu is None or not gcu.is_available():
        return

    # torch_gcu 2.11 bundles a flag_gems whose aten::exponential_ Python
    # kernel lacks the ``generator`` argument that vLLM's TopKTopP sampler
    # passes; install the generator-aware shim before any sampling runs.
    from vllm_fl.patches.exponential_compat import apply_exponential_compat

    apply_exponential_compat()
    apply_bilinear_pos_embed_gcu_patch()
    apply_causal_conv1d_gcu_patch()
    apply_chunk_delta_h_gcu_patch()
    apply_fla_autotune_gcu_patch()
    apply_fla_grid_gcu_patch()
    apply_sampler_cpu_detour()
    apply_penalties_gcu_patch()
    apply_logprob_gcu_patch()
    apply_fused_recurrent_packed_decode_gcu_patch()
    apply_kv_zeroer_gcu_patch()
    apply_mamba_copy_gcu_patch()
    apply_index_fill_gcu_patch()
    apply_inductor_gcu_patch()
    apply_vocab_mask_gcu_patch()
    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
