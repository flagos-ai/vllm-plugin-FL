# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems implementations of the EMPTY wheel's sparse-indexer top-k ops."""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels

_OP_NAMES = ("cooperative_topk", "persistent_topk")
# Keep the Library alive: its finalizer unregisters these operators.
# The next registration call reads this handle to avoid duplicate registration.
_lib: torch.library.Library | None = None


def _indexer_topk(logits, lengths, output, workspace, k, max_seq_len) -> None:
    from flag_gems import top_k_per_row_decode

    tensors = (logits, lengths, output, workspace)
    if any(not tensor.is_cuda or tensor.device != logits.device for tensor in tensors):
        raise ValueError("Indexer top-k tensors must share one CUDA device")
    if logits.dtype != torch.float32:
        raise ValueError("Indexer top-k logits must be float32")
    if lengths.dtype != torch.int32 or output.dtype != torch.int32:
        raise ValueError("Indexer top-k lengths and output must be int32")
    if workspace.dtype != torch.uint8:
        raise ValueError("Indexer top-k workspace must be uint8")
    if logits.ndim != 2 or logits.stride(1) != 1:
        raise ValueError("Indexer top-k logits must have contiguous rows")
    if lengths.ndim not in (1, 2) or not lengths.is_contiguous():
        raise ValueError("Indexer top-k lengths must be contiguous and 1D or 2D")
    if lengths.numel() != logits.shape[0]:
        raise ValueError("Indexer top-k lengths size mismatch")
    if k not in (512, 1024, 2048):
        raise ValueError("Indexer top-k supports k=512, 1024, or 2048")
    if output.shape != (logits.shape[0], k) or not output.is_contiguous():
        raise ValueError("Indexer top-k output must be contiguous with shape [rows, k]")
    if max_seq_len < 0:
        raise ValueError("Indexer top-k max_seq_len must be non-negative")
    if logits.shape[0] == 0:
        return

    # vLLM's persistent CUDA kernel clamps padded/invalid lengths before
    # selecting; Gems consumes lengths directly. Keep this device-side so
    # replay observes updated lengths without synchronization or host reads.
    row_bound = min(max_seq_len, logits.shape[1])
    valid_lengths = lengths.reshape(-1).clamp(min=0, max=row_bound)
    # next_n=1 means one independent length per row (including prefill QSA).
    # Both implementations write the unsorted selected indices in-place,
    # and emit 0..length-1 followed by -1 when length <= k. Equal boundary
    # scores may select any tied indices: native radix/atomic paths do not
    # provide a globally stable tie order either.
    top_k_per_row_decode(
        logits,
        1,
        valid_lengths,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k,
    )


def register_nvidia_topk_ops() -> bool:
    """Install executable CUDA kernels without overriding native vLLM builds."""
    global _lib
    if has_native_vllm_kernels():
        return False
    if _lib is not None:
        return True
    lib = torch.library.Library("_C", "FRAGMENT")
    for name in _OP_NAMES:
        if not hasattr(torch.ops._C, name):
            lib.define(
                f"{name}(Tensor logits, Tensor lengths, Tensor! output, "
                "Tensor workspace, int k, int max_seq_len) -> ()"
            )
        if not torch._C._dispatch_has_kernel_for_dispatch_key(f"_C::{name}", "CUDA"):
            lib.impl(name, _indexer_topk, "CUDA")
    _lib = lib
    return True
