# Copyright (c) 2026 BAAI. All rights reserved.
"""vLLM compressed-tensors adapter for the plugin-local WNA16 MoE."""

from __future__ import annotations

from importlib import import_module

import torch

from . import kernels


_ADAPTER_MARKER = "_vllm_fl_local_wna16_moe"
_UPSTREAM_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16"
)


def _build_local_moe_method(base_method):
    class FLCompressedTensorsWNA16MoEMethod(base_method):
        """Keep vLLM weight loading but call the fixed plugin operator."""

        _vllm_fl_local_wna16_moe = True

        def apply(
            self,
            layer,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del shared_experts_input
            return kernels.wna16_moe(
                x=x,
                w13_weight_packed=layer.w13_weight_packed,
                w2_weight_packed=layer.w2_weight_packed,
                w13_weight_scale=layer.w13_weight_scale,
                w2_weight_scale=layer.w2_weight_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                num_bits=self.num_bits,
                group_size=self.group_size,
                activation=layer.activation,
                apply_router_weight_on_input=(
                    layer.apply_router_weight_on_input
                ),
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                inplace=not self.moe.disable_inplace,
            )

    return FLCompressedTensorsWNA16MoEMethod


def install_fl_wna16_moe_method() -> bool:
    """Use the local MoE adapter when the fixed plugin operator is built."""
    if not kernels.is_wna16_moe_available():
        return False

    module = import_module(_UPSTREAM_MODULE)
    current = module.CompressedTensorsWNA16MoEMethod
    if getattr(current, _ADAPTER_MARKER, False):
        return True
    module.CompressedTensorsWNA16MoEMethod = _build_local_moe_method(current)
    return True


__all__ = ["install_fl_wna16_moe_method"]
