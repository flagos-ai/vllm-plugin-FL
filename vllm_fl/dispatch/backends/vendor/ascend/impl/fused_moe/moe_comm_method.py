# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from .moe_mlp import unified_apply_mlp
from .prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    QuantType,
)
from .token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


_MoECommMethods: Dict[Optional[MoECommType], MoECommMethod] = {}


def get_moe_comm_method(
    moe_comm_type: Optional[MoECommType],
) -> Optional[MoECommMethod]:
    return _MoECommMethods.get(moe_comm_type, None)


def setup_moe_comm_method(moe_config):
    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    return True


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int | None = None
    expert_tokens: torch.Tensor | None = None


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states, router_logits, mc2_mask, context_metadata = self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )
        return hidden_states, router_logits, mc2_mask, context_metadata

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        context_metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(
            hidden_states, reduce_results, context_metadata
        )
        return hidden_states

    def fused_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | list[torch.Tensor],
        w2: torch.Tensor | list[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_int8_w8a8: bool = False,
        use_int4_w4a8: bool = False,
        use_int4_w4a16: bool = False,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[list[torch.Tensor]] = None,
        w2_scale: Optional[list[torch.Tensor]] = None,
        w1_scale_bias: torch.Tensor = None,
        w2_scale_bias: torch.Tensor = None,
        w1_offset: Optional[torch.Tensor] = None,
        w2_offset: Optional[torch.Tensor] = None,
        # For load balance
        log2phy: torch.Tensor = None,
        need_trans: bool = False,
        dynamic_eplb: bool = False,
        mc2_mask: torch.Tensor = None,
        pertoken_scale: Optional[torch.Tensor] = None,
    ):
        # Check constraints
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
        ]

        before_dispatch_evt = torch.npu.current_stream().record_event()
        # Apply log2phy if needed
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        dispatch_results = self.token_dispatcher.token_dispatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            global_redundant_expert_num=self.moe_config.global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_quant=use_int8_w8a8 or use_int4_w4a8,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale,
        )

        mlp_output = unified_apply_mlp(
            hidden_states=dispatch_results.hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=dispatch_results.group_list,
            dynamic_scale=dispatch_results.dynamic_scale,
            group_list_type=dispatch_results.group_list_type,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
            topk_scales=dispatch_results.topk_scales,
            with_quant=use_int8_w8a8 or use_int4_w4a8 or use_int4_w4a16,
            fusion=use_int8_w8a8 and self.use_fusion_ops,
            need_trans=need_trans,
            dynamic_eplb=dynamic_eplb,
        )

        before_combine_evt = torch.npu.current_stream().record_event()
        combine_results = self.token_dispatcher.token_combine(
            hidden_states=mlp_output, context_metadata=dispatch_results.context_metadata
        )

        return FusedExpertsResult(
            routed_out=combine_results.routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=dispatch_results.group_list_type,
            expert_tokens=dispatch_results.group_list,
        )

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)
