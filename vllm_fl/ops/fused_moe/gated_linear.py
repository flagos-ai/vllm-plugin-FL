import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm_fl.dispatch import CachedOp

_router_gemm_bf16_fp32 = CachedOp("router_gemm_bf16_fp32")

class GateLinearFL(GateLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        import vllm._custom_ops as ops

        # Tier 1: DSV3 specialized kernel
        if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 2: cuBLAS bf16->fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = _router_gemm_bf16_fp32( x, self.weight)
            return output, None

        # Tier 3: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
