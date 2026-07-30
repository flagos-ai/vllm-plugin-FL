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
"""vLLM MPLinearKernel adapter for the plugin-local WNA16 GEMM."""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types

from . import kernels


class FLWNA16LinearKernel(MPLinearKernel):
    """Consume standard uint4b8 weights through the fixed local operator.

    vLLM's MPLinear registry selects this adapter only when
    ``vllm_fl::wna16_gemm`` is registered. The execution path does not fall
    through to CUDA, FlagGems, or the test-only reference implementation.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(
        cls,
        config: MPLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if config.weight_type != scalar_types.uint4b8:
            return False, "FL WNA16 currently requires symmetric uint4b8"
        if config.zero_points:
            return False, "FL WNA16 does not use explicit zero points"
        if config.has_g_idx:
            return False, "FL WNA16 does not support activation ordering"
        if config.act_type not in {torch.bfloat16, torch.float16}:
            return False, "FL WNA16 requires BF16 or FP16 activations"
        if config.group_size <= 0:
            return False, "FL WNA16 requires group-wise quantization"
        input_size, output_size = config.partition_weight_shape
        if input_size % config.group_size:
            return False, "input size must be divisible by group_size"
        if input_size % 8:
            return False, "input size must be divisible by the INT4 pack factor"
        if output_size <= 0:
            return False, "output size must be positive"
        if not kernels.is_wna16_gemm_available():
            return False, "the plugin-local wna16_gemm kernel is not built"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scale, _, _ = self._get_weight_params(layer)
        if weight.dtype != torch.int32 or weight.ndim != 2:
            raise ValueError("FL WNA16 expects weight_packed as 2D int32")
        if not weight.is_contiguous():
            weight.data = weight.data.contiguous()
        if not scale.is_contiguous():
            scale.data = scale.data.contiguous()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight, scale, _, _ = self._get_weight_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        output = kernels.wna16_gemm(
            x_2d,
            weight,
            scale,
            self.config.group_size,
            bias,
        )
        return output.reshape(*original_shape[:-1], output.shape[-1])


def register_fl_wna16_linear_kernel(registry: dict) -> bool:
    """Prepend the FL kernel only when its fixed local operator is built."""
    if not kernels.is_wna16_gemm_available():
        return False
    from vllm.platforms import PlatformEnum

    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLWNA16LinearKernel not in candidates:
        candidates.insert(0, FLWNA16LinearKernel)
    return True


__all__ = ["FLWNA16LinearKernel", "register_fl_wna16_linear_kernel"]
