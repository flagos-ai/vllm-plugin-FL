# Copyright (c) 2026 BAAI. All rights reserved.
"""Expose MiniMax M3's fused projections to checkpoint quantization matching.

Re-export the upstream classes themselves so multimodal processor registration
and model execution stay unchanged. The paired vLLM 0.24 classes omit this
metadata, causing named quantization targets to fall through to plain linears.
"""

from vllm.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)

from .minimax_m3_registration import add_fused_quantization_mapping

for _model_cls in (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
):
    add_fused_quantization_mapping(_model_cls)


__all__ = ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"]
