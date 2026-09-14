# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-gated NVIDIA fast paths for Qwen3.8-Flash-Next.

The model remains importable and runnable without NVIDIA extensions.  Every
entry point below first checks the active vLLM platform and, for private C++
operators, verifies that a CUDA dispatch kernel is registered.  This avoids
mistaking plugin schema stubs for an executable NVIDIA implementation.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable

import torch
from torch import nn

from vllm.platforms import current_platform


def is_nvidia_platform() -> bool:
    """Return true only for vLLM's NVIDIA CUDA platform, never ROCm."""

    try:
        return bool(current_platform.is_cuda()) and not bool(
            current_platform.is_rocm()
        )
    except (AttributeError, RuntimeError):
        return False


def _has_cuda_kernel(qualified_op: str) -> bool:
    if not is_nvidia_platform():
        return False
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified_op, "CUDA"
            )
        )
    except (AttributeError, RuntimeError):
        return False


def has_native_topk() -> bool:
    """Both branches are required because rows > 32 use persistent TopK."""

    return _has_cuda_kernel("_C::cooperative_topk") and _has_cuda_kernel(
        "_C::persistent_topk"
    )


def native_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    """Run the vLLM 0.24 native TopK ABI after capability selection."""

    if not has_native_topk():
        raise RuntimeError("NVIDIA native QSA TopK is unavailable")
    use_cooperative = (
        output.shape[0] <= 32
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    op = (
        torch.ops._C.cooperative_topk
        if use_cooperative
        else torch.ops._C.persistent_topk
    )
    op(logits, lengths, output, workspace, k, max_seq_len)


def has_native_cache_update() -> bool:
    return _has_cuda_kernel("_C_cache_ops::reshape_and_cache_flash")


def native_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Store K/V with vLLM's fused FlashAttention cache-update kernel."""

    if not has_native_cache_update():
        raise RuntimeError("NVIDIA fused QSA cache update is unavailable")
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


@lru_cache(maxsize=1)
def _flashinfer_gemma_rmsnorm() -> Callable[..., torch.Tensor] | None:
    if not is_nvidia_platform():
        return None
    try:
        from flashinfer.norm import gemma_rmsnorm
    except (ImportError, OSError):
        return None
    return gemma_rmsnorm


def fast_gemma_rmsnorm(
    norm: nn.Module,
    tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Return a FlashInfer result on NVIDIA, or ``None`` for fallback."""

    fn = _flashinfer_gemma_rmsnorm()
    if fn is None or tensor.device.type != "cuda":
        return None
    return fn(tensor, norm.weight, norm.variance_epsilon)


@lru_cache(maxsize=1)
def _mrope_kernel() -> tuple[Callable[..., Any] | None, bool]:
    if not is_nvidia_platform():
        return None, False
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            triton_mrope,
        )
    except (ImportError, OSError):
        return None, False
    try:
        has_neox_argument = "is_neox_style" in inspect.signature(
            triton_mrope
        ).parameters
    except (TypeError, ValueError):
        # vLLM 0.24 is the known eight-argument ABI. Unknown wrappers should
        # stay on the public/native fallback rather than be guessed here.
        return None, False
    return triton_mrope, has_neox_argument


def fast_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """Apply the measured NVIDIA 1-D/MRoPE kernels, with ABI probing."""

    if not is_nvidia_platform() or tensor.device.type != "cuda":
        return None
    num_tokens, _, head_dim = tensor.shape
    rotary_dim = int(rotary_emb.rotary_dim)
    if positions.ndim == 2:
        fn, has_neox_argument = _mrope_kernel()
        if fn is None:
            return None
        args = (
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
        )
        if has_neox_argument:
            rotated, _ = fn(*args, rotary_emb.is_neox_style)
        else:
            rotated, _ = fn(*args)
        return rotated.reshape_as(tensor)

    apply_rotary = getattr(rotary_emb, "apply_rotary_emb", None)
    forward_cuda = getattr(apply_rotary, "forward_cuda", None)
    if not callable(forward_cuda):
        return None
    rotated = forward_cuda(tensor[..., :rotary_dim], cos, sin)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


__all__ = [
    "fast_gemma_rmsnorm",
    "fast_qsa_rope",
    "has_native_cache_update",
    "has_native_topk",
    "is_nvidia_platform",
    "native_cache_update",
    "native_topk",
]
