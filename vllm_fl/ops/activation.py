import torch
from flag_gems.modules.activation import gems_silu_and_mul
from vllm.model_executor.layers.activation import SiluAndMul


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return gems_silu_and_mul(x1, x2)


__all__ = ["SiluAndMulFL"]
