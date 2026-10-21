# Copyright (c) 2026 BAAI. All rights reserved.

"""Implement the v0.32 padded MoE softmax router for NVIDIA EMPTY wheels."""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels

# Keep the Library alive: its finalizer unregisters these operators.
# The next registration call reads this handle to avoid duplicate registration.
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


def _moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    maybe_expert_map,
    scatter_idx=None,
) -> None:
    from vllm_fl.ops.nvidia_moe_align import moe_align_block_size_nvidia

    buffers = (sorted_token_ids, experts_ids, num_tokens_post_pad)
    if any(
        t.device != topk_ids.device or t.dtype != torch.int32 or not t.is_contiguous()
        for t in buffers
    ):
        raise ValueError("MoE alignment outputs must be contiguous CUDA int32 buffers")
    if (
        sorted_token_ids.ndim != 1
        or experts_ids.ndim != 1
        or num_tokens_post_pad.shape != (1,)
    ):
        raise ValueError("MoE alignment output buffer shapes are invalid")
    if scatter_idx is not None and (
        scatter_idx.device != topk_ids.device
        or scatter_idx.dtype != torch.int32
        or scatter_idx.shape != (topk_ids.numel(), 1)
    ):
        raise ValueError("MoE scatter indices must be a CUDA int32 route column")

    aligned = moe_align_block_size_nvidia(
        topk_ids, block_size, num_experts, maybe_expert_map
    )
    if (
        sorted_token_ids.numel() < aligned[0].numel()
        or experts_ids.numel() < aligned[1].numel()
    ):
        raise ValueError("MoE alignment output buffer capacity is too small")
    sorted_token_ids.fill_(topk_ids.numel())
    experts_ids.fill_(-1)
    sorted_token_ids[: aligned[0].numel()].copy_(aligned[0])
    experts_ids[: aligned[1].numel()].copy_(aligned[1])
    num_tokens_post_pad.copy_(aligned[2])
    if scatter_idx is not None:
        ids = topk_ids.reshape(-1).long()
        valid = (ids >= 0) & (ids < num_experts)
        if maybe_expert_map is not None:
            local_ids = maybe_expert_map[ids.clamp(0, num_experts - 1)]
            valid &= (local_ids >= 0) & (local_ids < num_experts)
        rows = torch.arange(ids.numel(), device=ids.device, dtype=torch.int32)
        scatter_idx.copy_(torch.where(valid, rows, -1)[:, None])


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
    if not hasattr(torch.ops._moe_C, "moe_align_block_size"):
        lib.define(
            "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
            "Tensor! sorted_token_ids, Tensor! experts_ids, Tensor! num_tokens_post_pad, "
            "Tensor? maybe_expert_map, Tensor(a!)? scatter_idx=None) -> ()"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_moe_C::moe_align_block_size", "CUDA"
    ):
        lib.impl("moe_align_block_size", _moe_align_block_size, "CUDA")
    _lib = lib
    return True
