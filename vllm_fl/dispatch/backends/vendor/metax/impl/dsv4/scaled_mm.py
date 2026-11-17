# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm_fl.dispatch.backends.flaggems.impl.deepseek_v4 import scaled_int8_quant
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4 import mctlass_ops


class MctlassInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if mctlass_ops.mctlass_op is None:
            return False, "MCTlass Python extension is unavailable"
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not isinstance(config, Int8ScaledMMLinearLayerConfig):
            return False, "Expected an INT8 scaled-MM layer configuration"
        if any(
            type(getattr(config, key, None)) is not bool
            for key in ("is_channelwise", "is_static_input_scheme", "input_symmetric")
        ):
            return False, "Unsupported INT8 quantization configuration"
        return cls.is_supported()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        x_q, x_s, x_zp = scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=azp_adj is None
        )

        if x_zp is not None:
            azp = None if i_zp is not None else x_zp
            return mctlass_ops.scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return mctlass_ops.scaled_mm(
            x_q,
            w_q,
            scale_a=x_s,
            scale_b=w_s,
            out_dtype=x.dtype,
            bias=bias,
        )
