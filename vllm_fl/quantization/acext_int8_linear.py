# Copyright (c) 2026 BAAI. All rights reserved.
"""acext INT8 W8A8 scaled-MM linear kernel for T-Head PPU.

Ported from the vendor fork's PPUInt8ScaledMMLinearKernel
(model_executor/kernels/linear/scaled_mm/ppu.py), acext branch only:
  - weights stay ROW-major (acext takes [N, K] int8 directly, unlike
    cutlass/triton kernels which transpose to [K, N])
  - activations quantized per-token via _C::dynamic_scaled_int8_quant
    (bridged to triton by vllm_fl) or static per-tensor scale
  - out = acext.int8_gemm(a_q, w_q, w_scale, a_scale, bias, out_dtype)

Only the symmetric path is implemented (this checkpoint is symmetric
per-channel weight / per-token activation); asymmetric falls back to
can_implement=False so the oracle picks the triton kernel instead.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

try:
    from acext import int8_gemm as _acext_int8_gemm
except ImportError:
    _acext_int8_gemm = None


def _w8a8_int8_matmul_acext(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert (bias is None) or (bias.shape[0] == b.shape[0] and bias.dtype == out_dtype)
    return _acext_int8_gemm(a, b, scale_b, scale_a, bias, out_dtype)


def _w8a8_int8_matmul_acext_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=out_dtype)


if _acext_int8_gemm is not None:
    direct_register_custom_op(
        op_name="fl_w8a8_int8_matmul_acext",
        op_func=_w8a8_int8_matmul_acext,
        fake_impl=_w8a8_int8_matmul_acext_fake,
    )


class AcextInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """acext int8 GEMM; row-major weights, symmetric quant only."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if _acext_int8_gemm is None:
            return False, "acext is not installed."
        if getattr(current_platform, "vendor_name", None) != "thead":
            return False, "requires T-Head PPU."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "acext kernel supports symmetric quantization only."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, _, i_s, _, _ = self._get_layer_params(layer)
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        # acext wants ROW-major [N, K] — keep the checkpoint layout as-is.
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.data, requires_grad=False),
        )

        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        if self.config.is_static_input_scheme:
            assert i_s is not None
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(i_s.max(), requires_grad=False),
            )
            setattr(layer, i_zp_name, None)
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        # dynamic per-token (i_s None) or static per-tensor quantization
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), i_s, i_zp, symmetric=True)

        out = torch.ops.vllm.fl_w8a8_int8_matmul_acext(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
        return out.view(*original_shape[:-1], out.shape[-1])
