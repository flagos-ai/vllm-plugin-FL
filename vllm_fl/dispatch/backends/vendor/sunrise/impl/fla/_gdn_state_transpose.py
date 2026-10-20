# Copyright (c) 2026 BAAI. All rights reserved.

"""Triton kernels to gather/transpose/scatter GDN state between vLLM and PTPU layouts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_DEFAULT_BLOCK_K: int = 64
_DEFAULT_BLOCK_V: int = 64


@triton.jit
def _fused_gather_transpose_kernel(
    src_ptr,            # initial_state: (num_blocks, HV, V, K), contig bf16/fp16/fp32
    idx_ptr,            # ssm_state_indices: (B,), int32
    dst_ptr,            # scratch: (>=B, HV, K, V), contig (same dtype as src)
    stride_src_n,       # initial_state.stride(0) = HV * V * K
    stride_src_hv,      # initial_state.stride(1) = V * K
    stride_src_v,       # initial_state.stride(2) = K
    stride_src_k,       # initial_state.stride(3) = 1
    stride_dst_b,       # scratch.stride(0) = HV * K * V
    stride_dst_hv,      # scratch.stride(1) = K * V
    stride_dst_k,       # scratch.stride(2) = V
    stride_dst_v,       # scratch.stride(3) = 1
    K,
    V,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program handles one ``(b, hv, k_block, v_block)`` tile.

    Reads ``initial_state[ssm_state_indices[b], hv, v_block, k_block]``
    of shape ``(BLOCK_V, BLOCK_K)`` (V outer, K inner), transposes to
    ``(BLOCK_K, BLOCK_V)``, and writes to
    ``scratch[b, hv, k_block, v_block]``.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_kv = tl.program_id(2)

    # Decompose the (k_block, v_block) program id. Using row-major
    # decomposition keeps adjacent ``pid_kv`` programs working on
    # adjacent ``v_block`` tiles, which on PTPU favours coalesced reads
    # of the K-innermost source layout (each ``v_block`` reads a full
    # K row stripe).
    num_v_blocks = tl.cdiv(V, BLOCK_V)
    pid_k = pid_kv // num_v_blocks
    pid_v = pid_kv % num_v_blocks

    k_start = pid_k * BLOCK_K
    v_start = pid_v * BLOCK_V

    k_offs = k_start + tl.arange(0, BLOCK_K)
    v_offs = v_start + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    # Slot for this batch row. int32 source → int64 for stride
    # arithmetic (state pools can be larger than 2^31 elements for
    # high-throughput configs).
    slot = tl.load(idx_ptr + pid_b).to(tl.int64)

    src_offsets = (
        slot * stride_src_n
        + pid_hv * stride_src_hv
        + v_offs[:, None] * stride_src_v
        + k_offs[None, :] * stride_src_k
    )
    src_mask = v_mask[:, None] & k_mask[None, :]
    src_tile = tl.load(src_ptr + src_offsets, mask=src_mask, other=0.0)

    # Transpose in-register: (BLOCK_V, BLOCK_K) → (BLOCK_K, BLOCK_V).
    # ``tl.trans`` is a metadata-only op on the in-SRAM tile; no extra
    # memory traffic.
    dst_tile = tl.trans(src_tile, 1, 0)

    # Store dst tile in (K, V) layout.
    dst_offsets = (
        pid_b * stride_dst_b
        + pid_hv * stride_dst_hv
        + k_offs[:, None] * stride_dst_k
        + v_offs[None, :] * stride_dst_v
    )
    dst_mask = k_mask[:, None] & v_mask[None, :]
    tl.store(dst_ptr + dst_offsets, dst_tile, mask=dst_mask)


@triton.jit
def _fused_transpose_scatter_kernel(
    src_ptr,            # scratch: (>=B, HV, K, V), contig (PTPU layout)
    idx_ptr,            # ssm_state_indices: (B,), int32
    dst_ptr,            # initial_state: (num_blocks, HV, V, K), contig (vLLM layout)
    stride_src_b,       # scratch.stride(0)
    stride_src_hv,      # scratch.stride(1)
    stride_src_k,       # scratch.stride(2)
    stride_src_v,       # scratch.stride(3)
    stride_dst_n,       # initial_state.stride(0)
    stride_dst_hv,      # initial_state.stride(1)
    stride_dst_v,       # initial_state.stride(2)
    stride_dst_k,       # initial_state.stride(3)
    K,
    V,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program handles one ``(b, hv, k_block, v_block)`` tile.

    Reads ``scratch[b, hv, k_block, v_block]`` of shape
    ``(BLOCK_K, BLOCK_V)`` (K outer, V inner), transposes to
    ``(BLOCK_V, BLOCK_K)``, and writes to
    ``initial_state[ssm_state_indices[b], hv, v_block, k_block]``.

    Padding-row skip
    ----------------
    Rows where ``slot == 0`` (vLLM's ``NULL_BLOCK_ID``) skip the
    writeback entirely. Matches FLA Triton's ``if state_idx <= 0:
      and prevents:

    1. Garbage state from cudagraph-padding rows polluting slot 0
       (the reserved NULL slot).
    2. Non-deterministic ``aten::index_put_``-style "last write wins"
       races between concurrent programs all targeting slot 0 -- with
       multiple padding rows this is a data race in Triton's
       across-program execution model.

    The unfused PyTorch ``initial_state[slot] = scratch.T(-1, -2)``
    chain wrote whatever ``scratch`` happens to contain for padding
    rows; in practice this was harmless because vLLM never reads slot 0,
    but it was also undefined-order-dependent. The fused kernel makes
    the behaviour explicit and deterministic.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_kv = tl.program_id(2)

    slot = tl.load(idx_ptr + pid_b).to(tl.int64)
    # Padding-row skip: NULL_BLOCK_ID == 0 in vLLM, plus defensive guard
    # against any future negative sentinel. Identical to FLA's
    # ``state_idx <= 0`` check.
    if slot <= 0:
        return

    num_v_blocks = tl.cdiv(V, BLOCK_V)
    pid_k = pid_kv // num_v_blocks
    pid_v = pid_kv % num_v_blocks

    k_start = pid_k * BLOCK_K
    v_start = pid_v * BLOCK_V

    k_offs = k_start + tl.arange(0, BLOCK_K)
    v_offs = v_start + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    # Load src tile from scratch: (BLOCK_K, BLOCK_V).
    src_offsets = (
        pid_b * stride_src_b
        + pid_hv * stride_src_hv
        + k_offs[:, None] * stride_src_k
        + v_offs[None, :] * stride_src_v
    )
    src_mask = k_mask[:, None] & v_mask[None, :]
    src_tile = tl.load(src_ptr + src_offsets, mask=src_mask, other=0.0)

    # Transpose: (BLOCK_K, BLOCK_V) → (BLOCK_V, BLOCK_K).
    dst_tile = tl.trans(src_tile, 1, 0)

    dst_offsets = (
        slot * stride_dst_n
        + pid_hv * stride_dst_hv
        + v_offs[:, None] * stride_dst_v
        + k_offs[None, :] * stride_dst_k
    )
    dst_mask = v_mask[:, None] & k_mask[None, :]
    tl.store(dst_ptr + dst_offsets, dst_tile, mask=dst_mask)


def gather_transpose_to_scratch(
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scratch: torch.Tensor,
    block_k: int = _DEFAULT_BLOCK_K,
    block_v: int = _DEFAULT_BLOCK_V,
) -> None:
    """Fused gather+transpose: copy slots out of vLLM's state pool into
    PTPU-layout scratch in one Triton launch.

    Equivalent to (but ~3 PyTorch launches less than):

    .. code-block:: python

        slot_idx.copy_(ssm_state_indices)             # int32 → int64
        scratch.copy_(initial_state[slot_idx].transpose(-1, -2))

    Args:
        initial_state: vLLM state pool, shape
            ``(num_blocks, HV, V, K)``, contiguous, any floating
            dtype.
        ssm_state_indices: per-sequence slot ids, shape ``(B,)``,
            ``int32``. ``0`` is vLLM's ``NULL_BLOCK_ID`` for padding
            rows (which read ``initial_state[0]`` -- guaranteed
            all-zero by vLLM convention).
        scratch: pre-allocated scratch in PTPU layout, shape
            ``(>=B, HV, K, V)``, contiguous, same dtype as
            ``initial_state``. Only the first ``B`` rows are written.
        block_k, block_v: Triton compile-time tile shape. Defaults
            ``(64, 64)`` give 4 tiles per ``(b, hv)`` for ``K = V =
            128``.
    """
    B = ssm_state_indices.shape[0]
    if B == 0:
        return

    HV, V, K = initial_state.shape[-3:]
    assert scratch.shape[-3] == HV, (
        f"scratch HV={scratch.shape[-3]} ≠ initial_state HV={HV}"
    )
    assert scratch.shape[-2] == K, (
        f"scratch K dim ({scratch.shape[-2]}) must equal "
        f"initial_state K ({K})"
    )
    assert scratch.shape[-1] == V, (
        f"scratch V dim ({scratch.shape[-1]}) must equal "
        f"initial_state V ({V})"
    )
    assert scratch.shape[0] >= B, (
        f"scratch capacity {scratch.shape[0]} < B={B}"
    )
    assert initial_state.dtype == scratch.dtype, (
        f"dtype mismatch: initial_state={initial_state.dtype}, "
        f"scratch={scratch.dtype}"
    )

    num_kv_blocks = triton.cdiv(K, block_k) * triton.cdiv(V, block_v)
    grid = (B, HV, num_kv_blocks)

    _fused_gather_transpose_kernel[grid](
        initial_state,
        ssm_state_indices,
        scratch,
        initial_state.stride(-4),
        initial_state.stride(-3),
        initial_state.stride(-2),
        initial_state.stride(-1),
        scratch.stride(-4),
        scratch.stride(-3),
        scratch.stride(-2),
        scratch.stride(-1),
        K,
        V,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )


def transpose_scatter_to_pool(
    scratch: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    initial_state: torch.Tensor,
    block_k: int = _DEFAULT_BLOCK_K,
    block_v: int = _DEFAULT_BLOCK_V,
) -> None:
    """Fused transpose+scatter: write PTPU-layout scratch back into
    vLLM's state pool in one Triton launch.

    Semantically equivalent to (but ~1 PyTorch launch less than):

    .. code-block:: python

        # NB: padding rows (slot_idx[b] == 0) DO NOT write back -- see
        # below.
        for b in range(B):
            if ssm_state_indices[b] > 0:
                initial_state[ssm_state_indices[b]] = (
                    scratch[b].transpose(-1, -2)
                )

    Args:
        scratch: PTPU-layout state, shape ``(>=B, HV, K, V)``,
            contiguous.
        ssm_state_indices: per-sequence slot ids, shape ``(B,)``,
            ``int32``. Rows with ``slot <= 0`` (vLLM's NULL_BLOCK_ID
            for padding) are **skipped** by the kernel -- matches
            FLA Triton's ``state_idx <= 0`` skip path and avoids
            non-deterministic races on slot 0.
        initial_state: vLLM state pool, shape ``(num_blocks, HV, V,
            K)``, contiguous.
        block_k, block_v: Triton compile-time tile shape.
    """
    B = ssm_state_indices.shape[0]
    if B == 0:
        return

    HV, V, K = initial_state.shape[-3:]
    assert scratch.shape[-3] == HV
    assert scratch.shape[-2] == K
    assert scratch.shape[-1] == V
    assert scratch.shape[0] >= B
    assert initial_state.dtype == scratch.dtype

    num_kv_blocks = triton.cdiv(K, block_k) * triton.cdiv(V, block_v)
    grid = (B, HV, num_kv_blocks)

    _fused_transpose_scatter_kernel[grid](
        scratch,
        ssm_state_indices,
        initial_state,
        scratch.stride(-4),
        scratch.stride(-3),
        scratch.stride(-2),
        scratch.stride(-1),
        initial_state.stride(-4),
        initial_state.stride(-3),
        initial_state.stride(-2),
        initial_state.stride(-1),
        K,
        V,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )
