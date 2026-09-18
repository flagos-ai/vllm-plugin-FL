# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_fl.dispatch import CachedOp

gemma = CachedOp("m3_gemma")
swiglu = CachedOp("m3_swiglu")


class SiluAndMulWithClampFL(torch.nn.Module):
    def __init__(self, swiglu_limit, alpha=1.0, beta=0.0, **kwargs):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x):
        return swiglu(x, self.swiglu_limit, self.alpha, self.beta)


qknorm_rope_insert = CachedOp("m3_qknorm_rope_insert")

patch_embed = CachedOp("m3_patch_embed")
