# Copyright (c) 2026 BAAI. All rights reserved.

"""Install the guarded decode activation before vLLM creates CustomOp instances."""

import functools


def patch_shared_swiglu() -> None:
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    original = SiluAndMulWithClamp.forward_native
    if getattr(original, "_vllm_fl_hygon_shared_swiglu", False):
        return

    @functools.wraps(original)
    def forward(self, x):
        from ..impl.native_ops.shared_activation import try_shared_swiglu

        out = try_shared_swiglu(x, self.swiglu_limit, self.alpha, self.beta)
        return original(self, x) if out is None else out

    forward._vllm_fl_hygon_shared_swiglu = True
    forward._vllm_fl_original = original
    SiluAndMulWithClamp.forward_native = forward
