from typing import ClassVar
import torch

from vllm.utils.torch_utils import direct_register_custom_op

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledDynamicMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledDynamicMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import DeepGemmFp8BlockScaledMMKernel
from vllm.model_executor.utils import replace_parameter

from flag_gems import w8a8_block_fp8_matmul

def _flaggems_fp8_block_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype,
) -> torch.Tensor:
    return w8a8_block_fp8_matmul(input, weight, input_scale, weight_scale, block_size, output_dtype=output_dtype)

class FlagGemsFp8BlockScaledMMLinearKernel(Fp8BlockScaledDynamicMMLinearKernel):
    base_type: ClassVar[type[FlashInferFp8DeepGEMMDynamicBlockScaledKernel]] = (
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel
    )
    fallback_type: ClassVar[type[DeepGemmFp8BlockScaledMMKernel]] = (
        DeepGemmFp8BlockScaledMMKernel
    )

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.base: FlashInferFp8DeepGEMMDynamicBlockScaledKernel
        self.fallback: DeepGemmFp8BlockScaledMMKernel

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # FlagGems not support float8_e8m0
        # convert e8m0 to float32
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )

        # float8_e8m0fnu: .to(int32) does value conversion not bit reinterpret,
        # so use native .float() which handles E8M0 correctly
        if weight_scale.dtype == torch.float8_e8m0fnu:
            new_weight_scale = weight_scale.float()
        # uint8 / int8 raw E8M0 bytes: interpret as IEEE 754 exponent
        elif weight_scale.element_size() == 1:
            # Reconstruct float32 from exponent: set exponent bits in IEEE 754 float32
            # float32 = sign(1) + exponent(8) + mantissa(23)
            # E8M0 value stored as raw exponent byte -> float = 2^(byte - 127)
            exp_bits = weight_scale.to(torch.int32) << 23
            new_weight_scale = exp_bits.view(torch.float32)
        else:
            new_weight_scale = weight_scale
        replace_parameter(layer, scale_attr_name, new_weight_scale.data) 


    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        weight_group_shape = self.config.weight_quant_key.scale.group_shape
        group_shape = [weight_group_shape.row, weight_group_shape.col]
        return torch.ops.vllm.flaggems_fp8_block_gemm(A, B, As, Bs, group_shape, output_dtype=self.config.out_dtype)

def _flaggems_fp8_block_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
            input.shape[0], weight.shape[0], dtype=output_dtype, device=input.device
    )

direct_register_custom_op(
    "flaggems_fp8_block_gemm",
    _flaggems_fp8_block_gemm_impl,
    fake_impl=_flaggems_fp8_block_gemm_fake,
)

