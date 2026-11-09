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
"""Hardware scaled-mm adapters for dynamic-token/per-channel W8A8."""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import (
    is_oot_enabled,
    use_flaggems_op,
)

FLAGGEMS_W8A8_LINEAR_OP = "w8a8_dynamic_per_token_linear"

_dynamic_per_token_quant_int8 = CachedOp("dynamic_per_token_quant_int8")
_prepare_ascend_weight = CachedOp("w8a8_prepare_weight")
_ascend_dynamic_linear = CachedOp("w8a8_dynamic_linear")


class FLW8A8DynamicLinearKernel(Int8ScaledMMLinearKernel):
    """Use FlagGems scaled-mm with the vLLM dynamic-W8A8 contract.

    Runtime operands are ``A=[M,K] int8``, ``B=[K,N] int8``,
    ``scale_a=[M,1] fp32`` and ``scale_b=[N] fp32``.
    """

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not is_oot_enabled():
            return False, "FL W8A8 kernels are disabled"
        if not use_flaggems_op(FLAGGEMS_W8A8_LINEAR_OP):
            return False, "FlagGems W8A8 linear is disabled by policy"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: Int8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires per-channel weights"
        if config.is_static_input_scheme:
            return False, "requires dynamic input quantization"
        if not config.input_symmetric:
            return False, "requires symmetric INT8 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight = getattr(layer, w_q_name)
        weight_scale = getattr(layer, w_s_name)
        if weight.ndim != 2 or weight.dtype != torch.int8:
            raise ValueError("FL W8A8 expects checkpoint weight as 2D int8")
        if weight_scale.numel() != weight.shape[0]:
            raise ValueError("FL W8A8 requires one scale per output channel")

        # FlagGems scaled_mm consumes B in [K, N] layout. Keep scales as [N].
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight.t().contiguous(), requires_grad=False),
        )
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(
                weight_scale.reshape(-1).to(torch.float32).contiguous(),
                requires_grad=False,
            ),
        )
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from flag_gems import scaled_mm

        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = _dynamic_per_token_quant_int8(x_2d)
        output = scaled_mm(
            x_q,
            weight,
            x_scale,
            weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return output.reshape(*original_shape[:-1], weight.shape[1])


class AscendInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """Use upstream checkpoint parameters and native NPU dynamic W8A8 math.

    CompressedTensorsW8A8Int8 owns allocation and checkpoint loading. This
    kernel prepares its logical [N,K] weights once into native [K,N] layout.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        del compute_capability
        if (
            getattr(current_platform, "vendor_name", None) != "ascend"
            or current_platform.device_type != "npu"
        ):
            return False, "requires the Ascend NPU platform"
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires per-channel weights"
        if config.is_static_input_scheme:
            return False, "requires dynamic input quantization"
        if not config.input_symmetric:
            return False, "requires symmetric INT8 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight, scale, _, _, _ = self._get_layer_params(layer)
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("Expected W8A8 [N,K] int8 weight")
        if scale.shape != (weight.shape[0], 1):
            raise ValueError("Expected one W8A8 scale per output channel")
        if weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Large-output DSA-CP Linear splitting is not enabled"
            )
        weight_nz_mode = getattr(layer, "_fl_ascend_weight_nz_mode", 1)
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        prepared_weight, prepared_scale = _prepare_ascend_weight(
            weight.data.transpose(0, 1).contiguous(),
            scale.data.flatten(),
            weight_nz_mode,
        )
        getattr(layer, w_q_name).data = prepared_weight
        getattr(layer, w_s_name).data = prepared_scale
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight, scale, _, _, _ = self._get_layer_params(layer)
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16 input")
        if x.ndim < 2 or x.shape[-1] != weight.shape[0]:
            raise ValueError("Expected [..., K] activations matching prepared [K,N]")
        if bias is not None and bias.shape != (weight.shape[1],):
            raise ValueError("Expected bias [N]")
        shape = (*x.shape[:-1], weight.shape[1])
        return _ascend_dynamic_linear(
            x.reshape(-1, x.shape[-1]).contiguous(), weight, scale, bias
        ).reshape(shape)


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prefer native Ascend on NPU, retaining the existing other-vendor order."""
    from vllm.platforms import PlatformEnum

    # PlatformFL always selects through the OOT registry. On NVIDIA that list
    # has already inherited CUDA's native Cutlass/Triton candidates, so
    # prepending here keeps them available as ordered fallbacks.
    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)
    if (
        AscendInt8ScaledMMLinearKernel.is_supported()[0]
        and AscendInt8ScaledMMLinearKernel not in candidates
    ):
        candidates.insert(0, AscendInt8ScaledMMLinearKernel)
    return True


__all__ = [
    "AscendInt8ScaledMMLinearKernel",
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "register_fl_w8a8_linear_kernel",
]
