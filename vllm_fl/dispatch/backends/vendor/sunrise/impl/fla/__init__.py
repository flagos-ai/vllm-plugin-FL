# Copyright (c) 2026 BAAI. All rights reserved.

"""
PTPU-native wrappers for vLLM's flash-linear-attention (FLA) ops.

The wrappers here adapt the upstream ``vllm.model_executor.layers.fla.ops.*``
signatures (which return Triton/FLA tensors) to PTPU's
``torch_ptpu.sgl_kernel`` Gated DeltaNet kernel family. They are designed to
be drop-in replacements; the monkey-patch in
``vllm_fl.dispatch.backends.vendor.sunrise.patches.patch_fla_ops`` rebinds the upstream
module attributes to these wrappers at vendor import time.

See ``patches/patch_fla_ops`` for the full rationale and rebind set.
"""

from .chunk_fwd_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .fused_recurrent_packed_decode import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from .fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .wy_fast import recompute_w_u_fwd

__all__ = [
    "chunk_fwd_o",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "l2norm_fwd",
    "recompute_w_u_fwd",
    "solve_tril",
]
