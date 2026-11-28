# Copyright (c) 2026 BAAI. All rights reserved.

"""Pack FLA logical programs into GCU's bounded physical launch grid."""

import sys


def gcu_flat_grid(programs: int) -> tuple[int, int]:
    if programs <= 0:
        raise ValueError("FLA launch must contain at least one program")
    x = min(programs, 65535)
    y = (programs + x - 1) // x
    if y > 255:
        raise ValueError(f"FLA launch exceeds GCU grid capacity: {programs}")
    return x, y


def apply_fla_grid_gcu_patch() -> None:
    import vllm.third_party.flash_linear_attention.ops as ops
    import vllm.third_party.flash_linear_attention.ops.chunk as chunk
    import vllm.third_party.flash_linear_attention.ops.chunk_o as chunk_o
    import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as gating

    from .chunk_o_gcu import chunk_fwd_o
    from .fused_sigmoid_gating_gcu import fused_sigmoid_gating_delta_rule_update

    chunk_o.chunk_fwd_o = chunk_fwd_o
    chunk.chunk_fwd_o = chunk_fwd_o
    gating.fused_sigmoid_gating_delta_rule_update = (
        fused_sigmoid_gating_delta_rule_update
    )
    ops.fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update
    # A model already imported by another vendor patch retains its own binding.
    model = sys.modules.get("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    if model is not None:
        model.fused_sigmoid_gating_delta_rule_update = (
            fused_sigmoid_gating_delta_rule_update
        )
