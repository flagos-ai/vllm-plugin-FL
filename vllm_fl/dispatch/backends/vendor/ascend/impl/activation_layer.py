"""Current vLLM-Ascend activation OOT classes owned by FL."""

import torch
import torch_npu
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul


class AscendQuickGELU(QuickGELU):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fast_gelu(x)


class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x)
