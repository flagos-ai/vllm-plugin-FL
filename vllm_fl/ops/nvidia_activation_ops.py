# Copyright (c) 2026 BAAI. All rights reserved.

"""Real clamped SwiGLU CUDA implementation for NVIDIA EMPTY wheels."""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels

# Keep the Library alive: its finalizer unregisters these operators.
# The next registration call reads this handle to avoid duplicate registration.
_lib: torch.library.Library | None = None


def _silu_and_mul_with_clamp(
    out: torch.Tensor,
    x: torch.Tensor,
    limit: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    if not x.is_cuda or not out.is_cuda or out.device != x.device:
        raise ValueError("Clamped SwiGLU tensors must share one CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Clamped SwiGLU supports FP16, BF16, and FP32 inputs")
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("Clamped SwiGLU needs an even final input dimension")
    width = x.shape[-1] // 2
    if out.shape != (*x.shape[:-1], width) or out.dtype != x.dtype:
        raise ValueError("Clamped SwiGLU output must match input shape and dtype")
    if out.numel() == 0:
        return

    # Preserve this adapter's FP32-reference behavior: the public provider
    # rounds the limit to its input dtype (e.g. BF16 cannot represent 0.7).
    # This does not claim bitwise parity with upstream's scalar/packed CUDA
    # kernels, which also round some intermediate values to the input dtype.
    gate = x[..., :width].float()
    up = x[..., width:].float()
    if alpha == 1.0 and beta == 0.0:
        from flaggems_vllm import silu_and_mul_with_clamp_out

        # v0.2.0 accepts a Python scalar without a host-to-device tensor copy.
        silu_and_mul_with_clamp_out(gate, up, out, limit)
        return

    # The existing Gems fused API has no alpha/beta parameters. Preserve the
    # complete upstream formula for SwiGLU-OAI and other non-default callers.
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    out.copy_(gate * torch.sigmoid(alpha * gate) * (up + beta))


def register_nvidia_activation_ops() -> bool:
    """Register the CUDA ABI without replacing any native vLLM kernel."""
    global _lib
    if has_native_vllm_kernels():
        return False
    if _lib is not None:
        return True
    lib = torch.library.Library("_C", "FRAGMENT")
    if not hasattr(torch.ops._C, "silu_and_mul_with_clamp"):
        lib.define(
            "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, "
            "float alpha=1.0, float beta=0.0) -> ()"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::silu_and_mul_with_clamp", "CUDA"
    ):
        lib.impl("silu_and_mul_with_clamp", _silu_and_mul_with_clamp, "CUDA")
    _lib = lib
    return True
