# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.


"""Graph-visible native mbtopk custom op for DSV4 prefill topK.

vllm_fl::dsv4_prefill_topk_native(logits, row_starts, row_ends, indices, top_k) mutates
logits (invalid positions -> -inf) and indices (row-local int32 top-k).  It is
intentionally strict and is only meant for DSV4 sparse prefill k=512/dim=1.
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


def _enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_PREFILL_NATIVE_TOPK", "0") == "1"


@triton.jit
def _mask_invalid_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride0: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_row: tl.constexpr = tl.cdiv(N, BLOCK)
    row = pid // blocks_per_row
    bid = pid % blocks_per_row
    start = tl.load(row_starts_ptr + row)
    end = tl.load(row_ends_ptr + row)
    if start == 0 and end >= N:
        return
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < N) & ((offs < start) | (offs >= end))
    tl.store(logits_ptr + row * stride0 + offs, float("-inf"), mask=mask)


@triton.jit
def _postprocess_kernel(
    src_ptr,
    dst_ptr,
    row_starts_ptr,
    top_k: tl.constexpr,
    src_stride0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < top_k
    start = tl.load(row_starts_ptr + row)
    vals = tl.load(src_ptr + row * src_stride0 + offs, mask=m, other=0)
    rel = (vals - start).to(tl.int32)
    tl.store(dst_ptr + row * top_k + offs, rel, mask=m)


@triton.jit
def _fill_short_rows_kernel(
    dst_ptr, row_starts_ptr, row_ends_ptr, top_k: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    start = tl.load(row_starts_ptr + row)
    end = tl.load(row_ends_ptr + row)
    valid = tl.maximum(end - start, 0)
    is_short = valid <= top_k
    vals = tl.where(offs < valid, offs, -1).to(tl.int32)
    tl.store(dst_ptr + row * top_k + offs, vals, mask=(offs < top_k) & is_short)


def _check(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
) -> None:
    if not _enabled():
        raise RuntimeError(
            "prefill_topk_native called while VLLM_FL_METAX_PREFILL_NATIVE_TOPK != 1"
        )
    if logits.dim() != 2 or indices.dim() != 2:
        raise RuntimeError(
            f"prefill_topk_native expects 2D logits/indices, got {logits.shape}, {indices.shape}"
        )
    if logits.dtype is not torch.float32:
        raise RuntimeError(
            f"prefill_topk_native requires fp32 logits, got {logits.dtype}"
        )
    if (
        row_starts.dtype is not torch.int32
        or row_ends.dtype is not torch.int32
        or indices.dtype is not torch.int32
    ):
        raise RuntimeError(
            f"prefill_topk_native requires int32 row_starts/row_ends/indices, got {row_starts.dtype}/{row_ends.dtype}/{indices.dtype}"
        )
    if (
        not logits.is_cuda
        or not row_starts.is_cuda
        or not row_ends.is_cuda
        or not indices.is_cuda
    ):
        raise RuntimeError("prefill_topk_native requires CUDA tensors")
    if not logits.is_contiguous():
        raise RuntimeError(
            f"prefill_topk_native requires contiguous logits, stride={logits.stride()}"
        )
    if logits.stride(1) != 1:
        raise RuntimeError(
            f"prefill_topk_native requires dim=1 contiguous, stride={logits.stride()}"
        )
    if top_k != 512:
        raise RuntimeError(f"prefill_topk_native only supports k=512, got {top_k}")
    if indices.shape[0] != logits.shape[0] or indices.shape[1] < top_k:
        raise RuntimeError(
            f"indices shape {indices.shape} incompatible with logits {logits.shape}, k={top_k}"
        )
    if row_starts.numel() < logits.shape[0] or row_ends.numel() < logits.shape[0]:
        raise RuntimeError("row_starts/row_ends shorter than num rows")


def _prefill_topk_native_impl(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
) -> None:
    _check(logits, row_starts, row_ends, indices, top_k)
    M, N = logits.shape
    mask_bs = 1024 if N <= 4096 else 2048
    _mask_invalid_kernel[(M * triton.cdiv(N, mask_bs),)](
        logits, row_starts, row_ends, logits.stride(0), N, BLOCK=mask_bs, num_warps=4
    )
    # Explicit aten op avoids Python-level flag_gems.topk monkeypatch; graph/profile must verify this remains native mbtopk.
    _vals, top_idx = torch.ops.aten.topk.default(logits, top_k, 1, True, False)
    _postprocess_kernel[(M,)](
        top_idx,
        indices,
        row_starts,
        top_k=top_k,
        src_stride0=top_idx.stride(0),
        BLOCK=512,
        num_warps=4,
    )
    # Preserve fixed sparse-indexer semantics for short rows: select all valid relative positions then -1 padding.
    _fill_short_rows_kernel[(M,)](
        indices, row_starts, row_ends, top_k=top_k, BLOCK=512, num_warps=4
    )


try:

    @torch.library.custom_op(
        "vllm_fl::dsv4_prefill_topk_native", mutates_args=("logits", "indices")
    )
    def prefill_topk_native(
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
    ) -> None:
        return _prefill_topk_native_impl(logits, row_starts, row_ends, indices, top_k)

    @prefill_topk_native.register_fake
    def _prefill_topk_native_fake(
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
    ) -> None:
        return None
except Exception:
    raise
