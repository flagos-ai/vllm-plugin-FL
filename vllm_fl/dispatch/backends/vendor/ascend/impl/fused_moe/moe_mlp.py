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

from typing import Optional

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.triton_utils import HAS_TRITON


def cumsum_group_list(
    group_list: torch.Tensor,
    src_list_type: int,
    dst_list_type: int,
    active_num: int = 0,
    expert_num: int = 0,
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,),
            fill_value=active_num,
            dtype=group_list.dtype,
            device=group_list.device,
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def unified_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    w1_scale: Optional[list[torch.Tensor]] = None,
    w2_scale: Optional[list[torch.Tensor]] = None,
    dynamic_scale: torch.Tensor = None,
    group_list_type: int = 1,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: Optional[torch.Tensor] = None,
    w2_offset: Optional[torch.Tensor] = None,
    topk_scales: Optional[torch.Tensor] = None,
    with_quant: bool = False,
    fusion: bool = False,
    need_trans: bool = True,
    dynamic_eplb: bool = False,
) -> torch.Tensor:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states
