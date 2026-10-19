# Copyright (c) 2026 BAAI. All rights reserved.

import logging

from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.causal_conv1d import apply_causal_conv1d_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.runner_compat import apply_runner_compat_patches
from .impl.sampler import apply_sampler_cpu_detour
from .impl.scatter_compat import apply_scatter_int32_patches

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

    # GCU native ATen has no int64 ``add`` kernel (topsatenAdd aborts the
    # worker); rewrite the one int64 add in the shared ModelRunnerFL here
    # rather than editing that cross-vendor file.
    apply_runner_compat_patches()

    # GCU native scatter/scatter_/scatter_add_ silently write only half of the
    # elements when the index tensor is int64 (the sampled-token write-back of
    # a batch therefore loses its second half); narrow such indices to int32.
    apply_scatter_int32_patches()

    # Graph-mode (inductor + cudagraph) compatibility patches: keep the fixes
    # that would otherwise require editing site-packages (torch_gcu /
    # triton_gcu / flag_gems) inside the plugin instead.
    from .impl.graph_compat import apply_graph_mode_compat_patches

    apply_graph_mode_compat_patches()
    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
