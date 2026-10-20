# Copyright (c) 2026 BAAI. All rights reserved.

"""Execute upstream's unfused TP embedding path inside the EMPTY CUDA op."""

import torch
import torch.nn.functional as F

from vllm_fl.nvidia_compat import has_native_vllm_kernels

_lib: torch.library.Library | None = None


def _vocab_parallel_embedding(
    out,
    input_ids,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        get_masked_input_and_mask,
    )

    if any(
        not t.is_cuda or t.device != weight.device for t in (out, input_ids, weight)
    ):
        raise ValueError("TP embedding tensors must share one CUDA device")
    if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("TP embedding input must be a 1D int32 or int64 tensor")
    if weight.ndim != 2 or out.shape != (input_ids.numel(), weight.shape[1]):
        raise ValueError("TP embedding output must have shape [tokens, embedding_dim]")
    if out.dtype != weight.dtype:
        raise ValueError("TP embedding output and weight must have the same dtype")

    # CUDA selection in v0.32 enables the fused op for every unquantized TP
    # embedding without checking whether its native kernel exists. Reuse the
    # exact upstream non-fused mask/offset logic, including added LoRA tokens
    # and padding. All computation stays on-device and can be graph captured.
    masked_ids, invalid = get_masked_input_and_mask(
        input_ids,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    out.copy_(F.embedding(masked_ids.long(), weight))
    out.masked_fill_(invalid.unsqueeze(-1), 0)


def register_nvidia_embedding_ops() -> bool:
    """Supply a CUDA implementation without overriding native vLLM wheels."""
    global _lib
    if has_native_vllm_kernels():
        return False
    if _lib is not None:
        return True
    lib = torch.library.Library("_C", "FRAGMENT")
    if not hasattr(torch.ops._C, "vocab_parallel_embedding"):
        lib.define(
            "vocab_parallel_embedding(Tensor! out, Tensor input_ids, Tensor weight, "
            "int org_vocab_start_index, int org_vocab_end_index, "
            "int num_org_vocab_padding, int added_vocab_start_index, "
            "int added_vocab_end_index) -> ()"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::vocab_parallel_embedding", "CUDA"
    ):
        lib.impl("vocab_parallel_embedding", _vocab_parallel_embedding, "CUDA")
    _lib = lib
    return True
