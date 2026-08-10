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
"""Direct binding reserved for the plugin-owned fused WNA16 MoE kernel."""

from __future__ import annotations

import torch


def _resolve_wna16_moe():
    """Resolve the fixed torch operator exported by this plugin's extension."""
    try:
        return torch.ops.vllm_fl.wna16_moe.default
    except AttributeError:
        return None


def is_wna16_moe_available() -> bool:
    return _resolve_wna16_moe() is not None


def wna16_moe(
    x: torch.Tensor,
    w13_weight_packed: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_bits: int,
    group_size: int,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    inplace: bool,
) -> torch.Tensor:
    """Run the plugin-owned fused WNA16 MoE operator."""
    kernel = _resolve_wna16_moe()
    if kernel is None:
        raise RuntimeError(
            "vllm_fl::wna16_moe is not built; implement it under "
            "vllm_fl/quantization/wna16/kernels"
        )
    return kernel(
        x,
        w13_weight_packed,
        w2_weight_packed,
        w13_weight_scale,
        w2_weight_scale,
        topk_weights,
        topk_ids,
        num_bits,
        group_size,
        activation,
        apply_router_weight_on_input,
        global_num_experts,
        expert_map,
        inplace,
    )


__all__ = ["is_wna16_moe_available", "wna16_moe"]
