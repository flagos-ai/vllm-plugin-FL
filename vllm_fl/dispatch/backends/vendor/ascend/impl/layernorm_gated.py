# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
import torch

import triton
import triton.language as tl

from triton_ascend_kernels.utils import get_npu_vectorcore_num


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row: tl.constexpr,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    ngroups,
    group_size: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    iter_per_row: tl.constexpr,
    rows_per_prog: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    pid = tl.program_id(0)

    row_start = pid * rows_per_prog
    for grp_block_id in range(0, iter_per_row):
        col_off = grp_block_id * BLOCK_N * group_size + tl.arange(
            0, BLOCK_N * group_size
        )
        col_mask = col_off < N

        # Load weights and biases
        w = tl.load(W + col_off, mask=col_mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + col_off, mask=col_mask, other=0.0).to(tl.float32)

        for row_block_id in range(0, rows_per_prog, BLOCK_M):
            row_block_id = tl.multiple_of(row_block_id, BLOCK_M)
            if row_start + row_block_id < M:
                row_off = row_start + row_block_id + tl.arange(0, BLOCK_M)
                row_mask = row_off < M

                tot_mask = (row_off.to(tl.float32)[:, None] < M) & (
                    col_off.to(tl.float32)[None, :] < N
                )
                X_base = X + row_off[:, None] * stride_x_row + col_off[None, :]
                Y_base = Y + row_off[:, None] * stride_y_row + col_off[None, :]
                if HAS_Z:
                    Z_base = Z + row_off[:, None] * stride_z_row + col_off[None, :]

                rstd_col_off = grp_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
                rstd_offsets = row_off[:, None] * ngroups + rstd_col_off[None, :]
                rstd_mask = row_mask[:, None] & (rstd_col_off[None, :] < ngroups)

                x = tl.load(X_base, mask=tot_mask, other=0.0).to(tl.float32)

                if HAS_Z and not NORM_BEFORE_GATE:
                    z = tl.load(Z_base, mask=tot_mask, other=0.0).to(tl.float32)
                    x *= z * tl.sigmoid(z)

                x = x.reshape(BLOCK_M * BLOCK_N, group_size)
                # Compute mean and variance per row (reduce along axis 1)
                if not IS_RMS_NORM:
                    mean = tl.sum(x, axis=1) / group_size  # Shape: [BLOCK_M * BLOCK_N]
                    # Store mean for each row
                    tl.store(
                        Mean + rstd_offsets,
                        mean.reshape(BLOCK_M, BLOCK_N),
                        mask=rstd_mask,
                    )
                    # Broadcast mean back to 2D for subtraction
                    xbar = tl.where(
                        tot_mask.reshape(BLOCK_M * BLOCK_N, group_size),
                        x - mean[:, None],
                        0.0,
                    )
                    var = (
                        tl.sum(xbar * xbar, axis=1) / group_size
                    )  # Shape: [BLOCK_M * BLOCK_N]
                else:
                    xbar = tl.where(
                        tot_mask.reshape(BLOCK_M * BLOCK_N, group_size), x, 0.0
                    )
                    var = (
                        tl.sum(xbar * xbar, axis=1) / group_size
                    )  # Shape: [BLOCK_M * BLOCK_N]
                    mean = 0.0  # Placeholder for RMS norm

                rstd = tl.rsqrt(var + eps)

                # Store rstd for each row
                tl.store(
                    Rstd + rstd_offsets, rstd.reshape(BLOCK_M, BLOCK_N), mask=rstd_mask
                )

                # Normalize and apply linear transformation
                if not IS_RMS_NORM:
                    x_hat = (x - mean[:, None]) * rstd[:, None]
                else:
                    x_hat = x * rstd[:, None]

                x_hat = x_hat.reshape(BLOCK_M, BLOCK_N * group_size)
                y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

                if HAS_Z and NORM_BEFORE_GATE:
                    z = tl.load(Z_base, mask=tot_mask, other=0.0).to(tl.float32)
                    y *= z * tl.sigmoid(z)

                # Write output
                tl.store(Y_base, y, mask=tot_mask)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1

    num_norm_blk = ngroups * M

    mean = (
        torch.empty((num_norm_blk,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((num_norm_blk,), dtype=torch.float32, device=x.device)

    # Tiling stage
    element_size = 4

    def get_resident_size(row_size):
        size_of_wgt = row_size * element_size
        size_of_bias = (row_size * element_size) if bias is not None else 0
        return size_of_wgt + size_of_bias

    def get_dyn_size(row_size):
        # Input
        size_of_x = row_size * element_size
        size_of_z = (row_size * element_size) if z is not None else 0

        # Intermidate
        size_of_sigz = (row_size * element_size * 2) if z is not None else 0
        size_of_mean = (row_size * element_size) if not is_rms_norm else 0
        size_of_xbar = row_size * element_size
        size_of_mask = row_size
        size_of_var = row_size * element_size
        size_of_hat = row_size * element_size

        return (
            size_of_x
            + size_of_z
            + size_of_sigz
            + size_of_mean
            + size_of_xbar
            + size_of_mask
            + size_of_var
            + size_of_hat
        )

    ub_size = 196608
    # 1. Row first
    grps_per_row = triton.cdiv(
        ub_size, get_resident_size(group_size) + get_dyn_size(group_size)
    )
    BLOCK_N = min(grps_per_row, ngroups)
    iter_per_row = triton.cdiv(ngroups, BLOCK_N)
    # 2. Col next
    core_num = get_npu_vectorcore_num()

    # For now, we dont handle small M, large N
    if M <= core_num:
        num_progs = M
        rows_per_prog = 1
        BLOCK_M = 1
    else:
        num_progs = core_num
        rows_per_prog = triton.cdiv(M, core_num)
        BLOCK_M = (
            (ub_size - get_resident_size(BLOCK_N * group_size))
            // get_dyn_size(BLOCK_N * group_size)
            // 2
        )

        BLOCK_M = triton.next_power_of_2(BLOCK_M) // 2
        BLOCK_M = max(BLOCK_M, 1)
        rows_per_prog = triton.cdiv(rows_per_prog, BLOCK_M) * BLOCK_M

    # Update grid to use rows_per_block
    grid = (num_progs,)
    layer_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        N,
        ngroups,
        group_size,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        iter_per_row=iter_per_row,
        rows_per_prog=rows_per_prog,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
    )
    return out, mean, rstd


def layer_norm_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    """Compatibility wrapper used by migrated Ascend layernorm code."""
    return layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )


__all__ = ["layer_norm_fwd", "layer_norm_fwd_npu"]
