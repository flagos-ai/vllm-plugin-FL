# Copyright (c) 2026 BAAI. All rights reserved.
"""Public compilation utilities and FX post-grad Inductor passes for vllm-plugin-FL.

Exports platform-aware CUDA Graph wrappers alongside composable post-grad FX
graph passes (`QuantFusionPass`, `RopeKvFusionPass`, `CompositePostGradPass`)
and the `register_post_grad_passes` helper used by vendor runtime hooks.
"""

from vllm_fl.compilation.break_graph import (
    BreakableCUDAGraphCapture,
    BreakableCUDAGraphWrapper,
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm_fl.compilation.graph import CompositePostGradPass, register_post_grad_passes
from vllm_fl.compilation.quant_fusion_pass import QuantFusionConfig, QuantFusionPass
from vllm_fl.compilation.rope_kv_fusion_pass import RopeKvFusionPass

__all__ = [
    "BreakableCUDAGraphCapture",
    "BreakableCUDAGraphWrapper",
    "CompositePostGradPass",
    "QuantFusionConfig",
    "QuantFusionPass",
    "RopeKvFusionPass",
    "eager_break_during_capture",
    "is_breakable_cudagraph_enabled",
    "register_post_grad_passes",
]
