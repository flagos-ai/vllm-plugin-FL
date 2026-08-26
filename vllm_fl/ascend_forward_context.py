# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend forward-context lifecycle owned by the FL runtime.

This is the Qwen3.6-relevant part of vLLM-Ascend 0.20.2's forward context.
FL keeps vLLM's base ``ForwardContext`` and attaches the NPU-only attributes
consumed by the copied attention, GDN and MoE implementations.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    *,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module | None = None,
    skip_compiled: bool = False,
    input_ids: torch.Tensor | None = None,
    ubatch_slices: Any = None,
    slot_mapping: Any = None,
):
    """Create vLLM's context and populate the Ascend execution contract."""
    with set_forward_context(
        attn_metadata,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=aclgraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        slot_mapping=slot_mapping,
        skip_compiled=skip_compiled,
    ):
        context = get_forward_context()
        # Qwen3.6 BF16 TP execution uses the all-gather/no-EP path.  The local
        # FL MoE implementation does not require vLLM-Ascend's communication
        # object, but copied operators still inspect these attributes.
        ascend_values = {
            "input_ids": input_ids,
            "num_tokens": num_tokens,
            "num_actual_tokens": (
                num_tokens if num_actual_tokens is None else num_actual_tokens
            ),
            "model_instance": model_instance,
            "capturing": False,
            "is_draft_model": False,
            "is_draft_model_prefill": False,
            "in_profile_run": False,
            "sinks": False,
            "layer_idx": None,
            "prefetch_mlp_gate_up_proj": False,
            "prefetch_mlp_down_proj": False,
            "moe_comm_type": MoECommType.ALLGATHER,
            "moe_comm_method": None,
            "mmrs_fusion": get_tensor_model_parallel_world_size() <= 8,
            "flash_comm_v1_enabled": False,
            "flashcomm_v2_enabled": False,
            "pad_size": 0,
            "padded_length": num_tokens,
            "padded_num_tokens": num_tokens,
            "max_tokens_across_dp": num_tokens,
            "max_tokens_across_pcp": num_tokens,
            "is_first_layer": True,
        }
        # Current copied code contains both direct ForwardContext reads and the
        # vLLM-Ascend _EXTRA_CTX proxy. Populate both stores so switching the
        # vLLM runner-generation flag cannot silently return None.
        for name, value in ascend_values.items():
            setattr(context, name, value)
        context.additional_kwargs.update(ascend_values)
        yield


class _ExtraForwardContextProxy:
    """Match vLLM-Ascend's ``_EXTRA_CTX`` access convention."""

    @staticmethod
    def _ctx():
        return get_forward_context()

    def __getattr__(self, name: str) -> Any:
        context = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            return context.additional_kwargs.get(name)
        return getattr(context, name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        context = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            context.additional_kwargs[name] = value
        else:
            setattr(context, name, value)


_EXTRA_CTX = _ExtraForwardContextProxy()
