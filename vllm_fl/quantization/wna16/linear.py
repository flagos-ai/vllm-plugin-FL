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

from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    TRITON_W4A16_SUPPORTED_GROUP_SIZES,
    TritonW4A16LinearKernel,
)

from . import kernels


class FLWNA16LinearKernel(MPLinearKernel):
    """Consume standard uint4b8/uint8b128 compressed-tensors weights.

    W4A16 uses the fixed plugin operator when it is built. W8A16 uses its
    dedicated operator when available and otherwise uses the portable Triton
    implementation.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(
        cls,
        config: MPLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if config.weight_type not in {
            scalar_types.uint4b8,
            scalar_types.uint8b128,
        }:
            return False, "FL WNA16 requires symmetric uint4b8 or uint8b128"
        if config.zero_points:
            return False, "FL WNA16 does not use explicit zero points"
        if config.has_g_idx:
            return False, "FL WNA16 does not support activation ordering"
        if config.act_type not in {torch.bfloat16, torch.float16}:
            return False, "FL WNA16 requires BF16 or FP16 activations"
        is_w8a16 = config.weight_type == scalar_types.uint8b128
        if config.group_size <= 0 and not (is_w8a16 and config.group_size == -1):
            return False, "FL WNA16 requires group or channel quantization"
        input_size, output_size = config.partition_weight_shape
        if config.group_size > 0 and input_size % config.group_size:
            return False, "input size must be divisible by group_size"
        pack_factor = 4 if is_w8a16 else 8
        if input_size % pack_factor:
            return False, "input size must be divisible by the pack factor"
        if output_size <= 0:
            return False, "output size must be positive"
        available = (
            kernels.is_w8a16_gemm_available()
            if is_w8a16
            else kernels.is_wna16_gemm_available()
        )
        if not available:
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
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        output = kernels.wna16_gemm(
            x_2d,
            weight,
            scale,
            self.config.group_size,
            bias,
            num_bits=self.config.weight_type.size_bits,
        )
        return output.reshape(*original_shape[:-1], output.shape[-1])

class HygonTritonWNA16LinearKernel(TritonW4A16LinearKernel):
    """Reuse vLLM's Triton W4A16 kernel on Hygon OOT devices."""

    @classmethod
    def can_implement(
        cls,
        config: MPLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if getattr(current_platform, "vendor_name", None) != "hygon":
            return False, "Hygon Triton WNA16 only targets Hygon"

        if config.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {config.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if config.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        output_size = config.partition_weight_shape[1]
        if output_size % 8 != 0:
            return (
                False,
                f"Output features ({output_size}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if config.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "HygonTritonWNA16LinearKernel",
            )

        group_size = config.group_size
        full_input_size = config.full_weight_shape[0]

        if (
            group_size != -1
            and group_size not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and group_size != full_input_size
        ):
            return (
                False,
                f"Group size {group_size} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({full_input_size})",
            )

        input_size = config.partition_weight_shape[0]
        effective_group_size = (
            group_size if group_size != -1 else input_size
        )

        if input_size % effective_group_size != 0:
            return (
                False,
                f"Input features {input_size} not divisible by "
                f"group size {effective_group_size}",
            )

        return True, None


def register_fl_wna16_linear_kernel(registry: dict) -> bool:
    """Prepend the FL kernel when a W4A16 or W8A16 backend is available."""
    from vllm.platforms import PlatformEnum, current_platform

    has_fl_kernel = (
        kernels.is_wna16_gemm_available()
        or kernels.is_w8a16_gemm_available()
    )
    is_hygon = getattr(current_platform, "vendor_name", None) == "hygon"

    if not (has_fl_kernel or is_hygon):
        return False

    candidates = registry.setdefault(PlatformEnum.OOT, [])

    # Hygon is represented as an OOT platform, but its
    # HIP Triton runtime can execute the upstream Triton W4A16 kernel.
    if is_hygon and HygonTritonWNA16LinearKernel not in candidates:
        candidates.insert(0, HygonTritonWNA16LinearKernel)

    # Prefer the plugin-local fixed operator when it is available.
    if has_fl_kernel and FLWNA16LinearKernel not in candidates:
        candidates.insert(0, FLWNA16LinearKernel)

    return True


__all__ = [
    "FLWNA16LinearKernel",
    "HygonTritonWNA16LinearKernel",
    "register_fl_wna16_linear_kernel",
]
