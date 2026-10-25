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
"""FlagGems scaled-mm adapter for dynamic-token/per-channel W8A8."""

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

# Capture the native INT8 GEMM before FlagGems' `enable()` monkeypatches
# ``torch._int_mm`` with its Triton implementation. That implementation cannot
# compile on Iluvatar (``tl.dot`` has no int8 x int8 -> int32 form), while the
# native C++ kernel is exact. Modules in ``vllm_fl`` are imported before the
# worker calls ``flag_gems.enable()``, so this reference is the untouched one.
_NATIVE_INT_MM = torch._int_mm


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
        from vllm.platforms import current_platform

        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = _dynamic_per_token_quant_int8(x_2d)

        if getattr(current_platform, "vendor_name", "") == "iluvatar":
            # Iluvatar: flag_gems.scaled_mm implements only the FP8 contract
            # ("mat1 and mat2 must be Float8 matrices"), and neither
            # torch._scaled_mm(int8) nor flag_gems.int_mm works here.
            # The native INT8 GEMM is the only viable path: it accumulates in
            # int32 (no overflow for int8 x int8), then we dequantize in fp32
            # with the per-token [M,1] and per-channel [N] scales.
            acc = torch._int_mm(x_q, weight)
            output = acc.to(torch.float32) * x_scale
            output = output * weight_scale.unsqueeze(0)
            if bias is not None:
                output = output + bias
            output = output.to(x.dtype)
        else:
            from flag_gems import scaled_mm

            output = scaled_mm(
                x_q,
                weight,
                x_scale,
                weight_scale,
                bias=bias,
                out_dtype=x.dtype,
            )
        return output.reshape(*original_shape[:-1], weight.shape[1])


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prepend FlagGems while retaining native INT8 kernels as fallback."""
    from vllm.platforms import PlatformEnum

    # PlatformFL always selects through the OOT registry. On NVIDIA that list
    # has already inherited CUDA's native Cutlass/Triton candidates, so
    # prepending here keeps them available as ordered fallbacks.
    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)
    return True


__all__ = [
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "register_fl_w8a8_linear_kernel",
]
