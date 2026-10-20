# Copyright (c) 2026 BAAI. All rights reserved.

"""Implement the v0.32 padded MoE softmax router for NVIDIA EMPTY wheels."""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels

_lib: torch.library.Library | None = None


def _topk_softmax(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    bias,
    is_padding,
) -> None:
    tensors = (topk_weights, topk_indices, token_expert_indices, gating_output)
    if any(not t.is_cuda or t.device != gating_output.device for t in tensors):
        raise ValueError("MoE router tensors must share one CUDA device")
    if gating_output.ndim != 2 or gating_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("MoE logits must be a 2D floating-point tensor")
    rows, experts = gating_output.shape
    if topk_weights.ndim != 2 or topk_weights.shape[0] != rows:
        raise ValueError("MoE router output must have shape [tokens, topk]")
    topk = topk_weights.shape[1]
    if not 0 < topk <= experts:
        raise ValueError("MoE topk must be between one and the number of experts")
    if (
        topk_weights.dtype != torch.float32
        or topk_indices.dtype not in (torch.int32, torch.int64, torch.uint32)
        or token_expert_indices.dtype != torch.int32
        or any(t.shape != topk_weights.shape for t in tensors[1:3])
        or any(not t.is_contiguous() for t in tensors[:3])
    ):
        raise ValueError("MoE router output shapes, dtypes or strides are invalid")
    if bias is not None and (
        bias.device != gating_output.device
        or bias.dtype != torch.float32
        or bias.shape != (experts,)
    ):
        raise ValueError("MoE correction bias must be a CUDA FP32 expert vector")
    if is_padding is not None and (
        is_padding.device != gating_output.device
        or is_padding.dtype != torch.bool
        or is_padding.shape != (rows,)
    ):
        raise ValueError("MoE padding mask must be a CUDA boolean token vector")
    if rows == 0:
        return

    if bias is None and topk <= 32 and experts <= 1024:
        from flag_gems import topk_softmax

        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output.contiguous(),
            renormalize,
        )
    else:
        # The correction bias affects selection only, not the routed weights.
        # Stable descending order matches the native kernel's lower-ID tie break.
        scores = torch.softmax(gating_output.float(), dim=-1)
        choice = scores if bias is None else scores + bias
        indices = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, :topk]
        weights = scores.gather(1, indices)
        if renormalize:
            denominator = weights.sum(-1, keepdim=True)
            weights = weights / torch.where(
                denominator > 0, denominator, torch.ones_like(denominator)
            )
        topk_weights.copy_(weights)
        topk_indices.copy_(indices)
        source_rows = (
            torch.arange(topk, device=gating_output.device, dtype=torch.int32)[None, :]
            * rows
            + torch.arange(rows, device=gating_output.device, dtype=torch.int32)[
                :, None
            ]
        )
        token_expert_indices.copy_(source_rows)

    if is_padding is not None:
        # v0.32 keeps the weights and source rows intact but sets every padded
        # expert ID to -1, so downstream alignment skips those tokens.
        if topk_indices.dtype == torch.uint32:
            topk_indices.copy_(topk_indices.long().masked_fill(is_padding[:, None], -1))
        else:
            topk_indices.masked_fill_(is_padding[:, None], -1)


def register_nvidia_moe_ops() -> bool:
    """Supply the missing CUDA op without overriding full native vLLM wheels."""
    global _lib
    if has_native_vllm_kernels():
        return False
    if _lib is not None:
        return True
    lib = torch.library.Library("_moe_C", "FRAGMENT")
    if not hasattr(torch.ops._moe_C, "topk_softmax"):
        lib.define(
            "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, "
            "Tensor! token_expert_indices, Tensor gating_output, bool renormalize, "
            "Tensor? bias, Tensor? is_padding) -> ()"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_moe_C::topk_softmax", "CUDA"
    ):
        lib.impl("topk_softmax", _topk_softmax, "CUDA")
    _lib = lib
    return True
