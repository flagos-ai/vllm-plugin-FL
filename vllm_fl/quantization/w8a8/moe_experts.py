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
"""vLLM-native experts adapter for dynamic per-token W8A8 MoE."""

from __future__ import annotations

from inspect import signature

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
)

try:
    # vLLM 0.24+
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
except ModuleNotFoundError as exc:
    if exc.name != "vllm.model_executor.layers.fused_moe.experts.triton_moe":
        raise
    # vLLM 0.20.x
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts,
    )

_FUSED_EXPERTS_HAS_INPLACE = "inplace" in signature(fused_experts).parameters


class TritonW8A8Experts(TritonExperts):
    """Run W8A8 MoE through vLLM's functional Triton implementation.

    The modular prepare stage must leave the first activation unquantized.
    ``fused_experts`` then owns both dynamic per-token quantization steps and
    consumes the checkpoint's per-channel INT8 weight scales.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del a2_scale, workspace13, workspace2, expert_tokens_meta

        if not self.quant_config.use_int8_w8a8:
            raise ValueError("TritonW8A8Experts requires an INT8 W8A8 quant config")
        if getattr(self, "_lora_context", None) is not None:
            raise NotImplementedError(
                "The vLLM functional W8A8 MoE adapter does not support LoRA"
            )
        if a1q_scale is not None:
            raise ValueError(
                "W8A8 activation was quantized before the functional experts "
                "adapter; expected an unquantized floating-point input"
            )
        if hidden_states.dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise TypeError(
                "W8A8 functional experts require floating-point hidden states, "
                f"got {hidden_states.dtype}"
            )

        if _FUSED_EXPERTS_HAS_INPLACE:
            result = fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=self.quant_config,
            )
        else:
            result = fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=self.quant_config,
            )
        output.copy_(result)


__all__ = [
    "TritonW8A8Experts",
]
