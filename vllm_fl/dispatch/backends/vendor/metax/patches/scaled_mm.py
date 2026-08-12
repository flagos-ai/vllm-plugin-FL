# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import PlatformEnum
from vllm_metax.model_executor.layers.quantization import _python_api_ops


class MctlassInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=azp_adj is None
        )

        if x_zp is not None:
            azp = None if i_zp is not None else x_zp
            return _python_api_ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return _python_api_ops.cutlass_scaled_mm(
            x_q,
            w_q,
            scale_a=x_s,
            scale_b=w_s,
            out_dtype=x.dtype,
            bias=bias,
        )


def register_mctlass_int8_kernel() -> None:
    linear._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = [
        MctlassInt8ScaledMMLinearKernel
    ]
