# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import triton
import triton.language as tl


@triton.jit
def mqa_hoist_skip_kernel(
    # Pointers to matrices
    q_ptr,
    kv_ptr,
    weights_ptr,
    cu_seq_len_k_start_ptr,
    cu_seq_len_k_end_ptr,
    output_ptr,
    # Matrix dimensions
    seq_len,
    seq_len_kv,
    num_heads,
    head_dim: tl.constexpr,
    # Strides
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvn,
    stride_kvd,
    stride_wm,
    stride_wh,
    stride_om,
    stride_on,
    # Options
    apply_mask: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fully fused MQA logits kernel in Triton.
    Computes: logits[m,n] = sum_h( ReLU(Q[m,h,:] @ KV[n,:]^T) * weights[m,h] )
    with optional masking.
    """
    tl.static_assert(head_dim == 128)
    tl.static_assert(BLOCK_K == 64)
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for valid positions
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len_kv

    if apply_mask:
        ks_bound = tl.load(
            cu_seq_len_k_start_ptr + offs_m, mask=mask_m, other=2147483647
        )
        ke_bound = tl.load(
            cu_seq_len_k_end_ptr + offs_m, mask=mask_m, other=-2147483648
        )
        lo = tl.min(ks_bound, axis=0)
        hi = tl.max(ke_bound, axis=0)
        if (pid_n * BLOCK_N >= hi) | ((pid_n + 1) * BLOCK_N <= lo):
            out_ptrs = (
                output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
            )
            tl.store(
                out_ptrs,
                tl.full((BLOCK_M, BLOCK_N), -float("inf"), tl.float32),
                mask=mask_m[:, None] & mask_n[None, :],
            )
            return

    dk = tl.arange(0, BLOCK_K)
    kp = kv_ptr + offs_n[:, None] * stride_kvn + dk[None, :] * stride_kvd
    kv0 = tl.load(kp, mask=mask_n[:, None], other=0.0)
    kv1 = tl.load(kp + BLOCK_K * stride_kvd, mask=mask_n[:, None], other=0.0)

    # Accumulator for all heads
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over heads
    for h in range(num_heads):
        # Load weights for this head
        w_ptrs = weights_ptr + offs_m * stride_wm + h * stride_wh
        w = tl.load(w_ptrs, mask=mask_m, other=0.0)

        # Accumulator for GEMM across K dimension
        gemm_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # GEMM: Q[m,h,:] @ KV[n,:]^T
        for k in range(0, head_dim, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < head_dim

            # Load Q tile: [BLOCK_M, BLOCK_K]
            q_ptrs = (
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + offs_k[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load KV tile: [BLOCK_N, BLOCK_K]
            kv_ptrs = (  # noqa: F841
                kv_ptr + offs_n[:, None] * stride_kvn + offs_k[None, :] * stride_kvd
            )
            kv = tl.where(k == 0, kv0, kv1)

            # Compute dot product: Q @ KV^T
            gemm_acc += tl.dot(q, tl.trans(kv))

        # Apply ReLU
        gemm_acc = tl.maximum(gemm_acc, 0.0)

        # Multiply by weight and accumulate
        acc += gemm_acc * w[:, None]

    # Apply mask if needed
    if apply_mask:
        start_ptrs = cu_seq_len_k_start_ptr + offs_m
        end_ptrs = cu_seq_len_k_end_ptr + offs_m
        start_idx = tl.load(start_ptrs, mask=mask_m, other=0)
        end_idx = tl.load(end_ptrs, mask=mask_m, other=seq_len_kv)

        # Check if each position is valid
        valid = (offs_n[None, :] >= start_idx[:, None]) & (
            offs_n[None, :] < end_idx[:, None]
        )
        acc = tl.where(valid, acc, float("-inf"))

    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
