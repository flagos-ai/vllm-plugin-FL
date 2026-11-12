# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from vllm_fl.ops.minimax_m3 import torch_ops

gemma = torch_ops.gemma
swiglu = torch_ops.swiglu


class SiluAndMulWithClampFL(torch.nn.Module):
    def __init__(self, swiglu_limit, alpha=1.0, beta=0.0, **kwargs):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x):
        return swiglu(x, self.swiglu_limit, self.alpha, self.beta)


qknorm_rope_insert = torch_ops.qknorm_rope_insert

patch_embed = torch_ops.patch_embed
