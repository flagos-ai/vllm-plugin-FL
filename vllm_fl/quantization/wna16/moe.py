# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM compressed-tensors adapter for the plugin-local WNA16 MoE."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import torch

from . import kernels

_ADAPTER_MARKER = "_vllm_fl_local_wna16_moe"
_UPSTREAM_MODULES = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16",
)


def is_flaggems_wna16_moe_available() -> bool:
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    return (
        is_oot_enabled()
        and use_flaggems_op("fused_moe")
        and find_spec("flag_gems") is not None
    )


def is_fl_wna16_moe_available() -> bool:
    return kernels.is_wna16_moe_available() or is_flaggems_wna16_moe_available()


def _build_local_moe_method(base_method):
    class FLCompressedTensorsWNA16MoEMethod(base_method):
        """Keep vLLM loading and route execution to an FL backend."""

        _vllm_fl_local_wna16_moe = True

        def select_gemm_impl(
            self,
            prepare_finalize,
            layer,
        ):
            if not is_flaggems_wna16_moe_available():
                return super().select_gemm_impl(
                    prepare_finalize,
                    layer,
                )
            from vllm_fl.ops.fused_moe.fused_moe_utils import (
                TritonExpertsFL,
            )

            layer.w13_weight = layer.w13_weight_packed
            layer.w2_weight = layer.w2_weight_packed
            return TritonExpertsFL(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

        def apply(
            self,
            layer,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            if not kernels.is_wna16_moe_available():
                return super().apply(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    shared_experts_input,
                )
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
                apply_router_weight_on_input=(layer.apply_router_weight_on_input),
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                inplace=not self.moe.disable_inplace,
            )

    return FLCompressedTensorsWNA16MoEMethod


def install_fl_wna16_moe_method() -> bool:
    """Install the local MoE method when an FL backend is available.

    vLLM 0.20.2 selects WNA16 MoE through a scheme class rather than a kernel
    registry, so replacing that class is the installation point.
    """
    if not is_fl_wna16_moe_available():
        return False

    module = None
    for module_name in _UPSTREAM_MODULES:
        try:
            module = import_module(module_name)
            current = module.CompressedTensorsWNA16MoEMethod
            break
        except (ImportError, AttributeError):
            continue
    if module is None:
        raise ImportError("Could not find vLLM's WNA16 MoE method")
    if getattr(current, _ADAPTER_MARKER, False):
        return True
    module.CompressedTensorsWNA16MoEMethod = _build_local_moe_method(current)
    return True


__all__ = [
    "install_fl_wna16_moe_method",
    "is_fl_wna16_moe_available",
    "is_flaggems_wna16_moe_available",
]
