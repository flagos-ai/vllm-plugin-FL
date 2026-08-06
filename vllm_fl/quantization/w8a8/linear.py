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

from importlib.util import find_spec

import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import (
    is_nvidia_platform,
    is_oot_enabled,
    use_flaggems_op,
)

FLAGGEMS_W8A8_LINEAR_OP = "w8a8_dynamic_per_token_linear"

_dynamic_per_token_quant_int8 = CachedOp("dynamic_per_token_quant_int8")


def _resolve_flaggems_scaled_mm():
    """Resolve the returning ``scaled_mm`` API, not the output-buffer API.

    FlagGems 5.3 exposes only ``cutlass_scaled_mm(output, ...)``.  Besides
    having a different call contract, its SM80 branch is not implemented on
    PPU.  The generic returning API was added later and provides the Triton
    INT8 fallback required by CUDA-alike OOT devices.
    """
    import flag_gems

    scaled_mm = getattr(flag_gems, "scaled_mm", None)
    if scaled_mm is None:
        try:
            from flag_gems.ops.scaled_mm import scaled_mm
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "FlagGems W8A8 linear requires the returning scaled_mm API"
            ) from exc
    if not callable(scaled_mm):
        raise RuntimeError("FlagGems scaled_mm is not callable")
    return scaled_mm


def _flaggems_available() -> bool:
    if find_spec("flag_gems") is None:
        return False
    try:
        _resolve_flaggems_scaled_mm()
    except (ImportError, RuntimeError):
        return False
    return True


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
        if is_nvidia_platform():
            return False, "NVIDIA keeps vLLM's native W8A8 kernels"
        if not is_oot_enabled():
            return False, "FL OOT kernels are disabled"
        if not use_flaggems_op(FLAGGEMS_W8A8_LINEAR_OP):
            return False, "FlagGems W8A8 linear is disabled by policy"
        if not _flaggems_available():
            return False, "FlagGems is not installed"
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
        scaled_mm = _resolve_flaggems_scaled_mm()

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


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prepend the FL kernel on non-NVIDIA OOT platforms."""
    if is_nvidia_platform() or not _flaggems_available():
        return False

    from vllm.platforms import PlatformEnum

    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)
    return True


__all__ = [
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "register_fl_w8a8_linear_kernel",
]
