# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""K3 projections plugged into the native MLA layer."""

import torch
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

from vllm_fl.models.kimi_k3_context import execution_config
from vllm_fl.dispatch import CachedOp

project_query = CachedOp("kimi_k3_query_projection")
from vllm_fl.ops.kimi_k3.projections import Tp8GateProjection


class K3MultiHeadLatentAttention(MultiHeadLatentAttention):
    gate_projection_cls = Tp8GateProjection

    def __init__(self, *args, **kwargs):
        self.k3_execution = execution_config()
        super().__init__(*args, **kwargs)

    def _project_query(self, query: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.k3_execution.tp8_compute:
            return project_query(query, weight)
        return super()._project_query(query, weight)
