# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM scaled-mm adapter for dynamic-token/per-channel W8A8."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from importlib.util import find_spec

import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from .reference import dynamic_per_token_quant_int8
from vllm_fl.utils import is_oot_enabled, use_flaggems_op

FLAGGEMS_W8A8_LINEAR_OP = "w8a8_dynamic_per_token_linear"
W8A8_LINEAR_BACKEND_ENV = "VLLM_FL_W8A8_LINEAR_BACKEND"
_VALID_W8A8_LINEAR_BACKENDS = {"auto", "flaggems", "vllm"}

logger = logging.getLogger(__name__)


def get_w8a8_linear_backend() -> str:
    """Return the requested backend for dynamic-token W8A8 linear layers."""
    backend = os.getenv(W8A8_LINEAR_BACKEND_ENV, "auto").strip().lower()
    if backend not in _VALID_W8A8_LINEAR_BACKENDS:
        choices = ", ".join(sorted(_VALID_W8A8_LINEAR_BACKENDS))
        raise ValueError(
            f"{W8A8_LINEAR_BACKEND_ENV} must be one of {choices}, got {backend!r}"
        )
    return backend


def _flaggems_available() -> bool:
    return find_spec("flag_gems") is not None


class FLW8A8DynamicLinearKernel(Int8ScaledMMLinearKernel):
    """Use FlagGems scaled-mm after dynamic per-token INT8 quantization."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
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

        # aten._scaled_mm consumes B in [K, N] layout. Keep scales as [N]
        # because FlagGems accepts a scalar or a vector of output-channel
        # scales.
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
        from flag_gems.ops.scaled_mm import scaled_mm

        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = dynamic_per_token_quant_int8(x_2d)
        output = scaled_mm(
            x_q,
            weight,
            x_scale,
            weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return output.reshape(*original_shape[:-1], weight.shape[1])


def _create_vllm_w8a8_linear_kernel(
    config: Int8ScaledMMLinearLayerConfig,
    layer_param_names: Sequence[str],
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    """Select a native vLLM INT8 kernel without selecting the FL adapter."""
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    candidates = [
        candidate
        for candidate in linear_module._POSSIBLE_INT8_KERNELS.get(
            current_platform._enum, []
        )
        if candidate is not FLW8A8DynamicLinearKernel
    ]
    if not candidates:
        raise RuntimeError(
            "vLLM has no native INT8 scaled-mm candidates for the current platform"
        )

    try:
        kernel_type = linear_module.choose_scaled_mm_linear_kernel(
            config,
            {current_platform._enum: candidates},
        )
    except ValueError as exc:
        raise RuntimeError(
            "no native vLLM INT8 scaled-mm kernel supports this platform"
        ) from exc

    logger.info(
        "Selected native vLLM W8A8 linear kernel %s for %s",
        kernel_type.__name__,
        module_name,
    )
    return kernel_type(config, layer_param_names)


def create_w8a8_linear_kernel(
    config: Int8ScaledMMLinearLayerConfig,
    layer_param_names: Sequence[str],
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    """Select FlagGems or a native vLLM W8A8 linear kernel.

    ``auto`` preserves the existing FlagGems-first behavior and only tries
    vLLM's native selector when FlagGems is unavailable. Explicit selections
    do not silently cross over to the other backend.
    """
    backend = get_w8a8_linear_backend()
    failures: list[str] = []

    if backend in {"auto", "flaggems"}:
        supported, reason = FLW8A8DynamicLinearKernel.is_supported()
        can_implement, implement_reason = FLW8A8DynamicLinearKernel.can_implement(
            config
        )
        if supported and can_implement:
            logger.info("Selected FlagGems W8A8 linear kernel for %s", module_name)
            return FLW8A8DynamicLinearKernel(config, layer_param_names)
        failures.append(reason or implement_reason or "FlagGems is unavailable")

    if backend in {"auto", "vllm"}:
        try:
            return _create_vllm_w8a8_linear_kernel(
                config,
                layer_param_names,
                module_name,
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    details = "; ".join(failures)
    raise RuntimeError(f"W8A8 linear backend {backend!r} is unavailable: {details}")


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prepend the FL W8A8 kernel when FlagGems is selectable."""
    if not _flaggems_available():
        return False
    from vllm.platforms import PlatformEnum

    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)
    return True


__all__ = [
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "W8A8_LINEAR_BACKEND_ENV",
    "create_w8a8_linear_kernel",
    "get_w8a8_linear_backend",
    "register_fl_w8a8_linear_kernel",
]
