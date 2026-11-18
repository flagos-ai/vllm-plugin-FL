# SPDX-License-Identifier: Apache-2.0
"""FP32 projection path for the 512-expert Hy4 Flash router."""

import torch

from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.platforms import current_platform

from vllm_fl.ops.hy4_flash_kernels import fp32_router_gemm


class Hy4FlashGateLinear(GateLinear):
    def forward(self, x):
        if (
            current_platform.is_device_capability((9, 0))
            and x.ndim == 2
            and x.shape[1] == 3072
            and x.dtype in (torch.bfloat16, torch.float32)
            and self.weight.shape == (512, 3072)
            and self.weight.dtype == torch.float32
        ):
            return self._return(fp32_router_gemm(x, self.weight))
        return super().forward(x)
