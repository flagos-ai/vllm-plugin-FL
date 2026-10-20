# Copyright (c) 2026 BAAI. All rights reserved.

"""Native PyTorch rotary provider for NVIDIA's kernel-free vLLM wheel."""

import torch

from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.platforms import current_platform

from vllm_fl.nvidia_compat import has_native_vllm_kernels


class NvidiaApplyRotaryEmb(ApplyRotaryEmb):
    """Keep upstream rotary semantics without importing bundled FlashAttention."""

    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)


def register_nvidia_rotary_ops() -> bool:
    """Install the EMPTY provider before model construction, preserving overrides."""
    if not current_platform.is_cuda() or has_native_vllm_kernels():
        return False

    registered = op_registry_oot.get("ApplyRotaryEmb")
    if registered is not None:
        return registered is NvidiaApplyRotaryEmb

    CustomOp.register_oot(name="ApplyRotaryEmb")(NvidiaApplyRotaryEmb)
    # The OOT lookup key is a class name; compilation +/- directives still
    # use the original operator name, just as they do for the upstream class.
    NvidiaApplyRotaryEmb.name = ApplyRotaryEmb.name
    return True
