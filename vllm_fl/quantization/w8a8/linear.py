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

_dynamic_per_token_quant_int8_op = CachedOp("dynamic_per_token_quant_int8")

def _flaggems_available() -> bool:
    return find_spec("flag_gems") is not None


def _resolve_flaggems_scaled_mm():
    """Resolve the returning ``scaled_mm`` API."""
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


# --- torch.library custom ops for graph mode compatibility ---
_resolved_quant_fn = None
_cached_scaled_mm = None


@torch.library.custom_op("vllm_fl::dynamic_per_token_quant_int8", mutates_args=())
def _dynamic_per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global _resolved_quant_fn
    if _resolved_quant_fn is None:
        from vllm_fl.dispatch import get_default_manager
        mgr = get_default_manager()
        impl = mgr._resolve_impl("dynamic_per_token_quant_int8")
        _resolved_quant_fn = impl.fn
    return _resolved_quant_fn(x)


@_dynamic_per_token_quant_int8.register_fake
def _dynamic_per_token_quant_int8_fake(x):
    M, K = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    return x_q, x_scale


@torch.library.custom_op("vllm_fl::scaled_mm", mutates_args=())
def _scaled_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: int,
) -> torch.Tensor:
    global _cached_scaled_mm
    if _cached_scaled_mm is None:
        _cached_scaled_mm = _resolve_flaggems_scaled_mm()
    dtype = _DTYPE_MAP[out_dtype]
    return _cached_scaled_mm(a, b, a_scale, b_scale, bias=None, out_dtype=dtype)


@_scaled_mm_wrapper.register_fake
def _scaled_mm_wrapper_fake(a, b, a_scale, b_scale, out_dtype):
    M = a.shape[0]
    N = b.shape[1]
    dtype = _DTYPE_MAP[out_dtype]
    return torch.empty((M, N), dtype=dtype, device=a.device)


# Encode dtype as int for custom_op (custom_op doesn't support torch.dtype args)
_DTYPE_MAP = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
}
_DTYPE_TO_INT = {v: k for k, v in _DTYPE_MAP.items()}


@torch.library.custom_op("vllm_fl::scaled_mm_bias", mutates_args=())
def _scaled_mm_bias_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: int,
) -> torch.Tensor:
    global _cached_scaled_mm
    if _cached_scaled_mm is None:
        _cached_scaled_mm = _resolve_flaggems_scaled_mm()
    dtype = _DTYPE_MAP[out_dtype]
    return _cached_scaled_mm(a, b, a_scale, b_scale, bias=bias, out_dtype=dtype)


@_scaled_mm_bias_wrapper.register_fake
def _scaled_mm_bias_wrapper_fake(a, b, a_scale, b_scale, bias, out_dtype):
    M = a.shape[0]
    N = b.shape[1]
    dtype = _DTYPE_MAP[out_dtype]
    return torch.empty((M, N), dtype=dtype, device=a.device)


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
        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = _dynamic_per_token_quant_int8(x_2d)
        dtype_int = _DTYPE_TO_INT[x.dtype]
        if bias is not None:
            output = _scaled_mm_bias_wrapper(
                x_q, weight, x_scale, weight_scale, bias, dtype_int)
        else:
            output = _scaled_mm_wrapper(
                x_q, weight, x_scale, weight_scale, dtype_int)
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
