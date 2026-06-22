# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
import torch
import numpy as np

import triton
import triton.language as tl

from triton_ascend_kernels.utils import get_npu_vectorcore_num


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    rows_per_prog: tl.constexpr,
    cols_per_row: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_BATCHES: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    prog_id, i_s = tl.program_id(0), tl.program_id(1)

    row_block_off = prog_id * rows_per_prog
    for i_d in range(0, cols_per_row):
        # Col direction
        head_off = i_d * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        head_mask = head_off < NUM_HEADS

        blk_A_log = tl.load(A_log + head_off, mask=head_mask)
        blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

        for row_id in range(0, rows_per_prog, BLOCK_BATCHES):
            row_id = tl.multiple_of(row_id, BLOCK_BATCHES)
            # Row direction
            i_b = row_block_off + row_id + tl.arange(0, BLOCK_BATCHES)
            batch_off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS
            batch_mask = i_b < NUM_BATCHES

            # 2D off & mask
            tot_off = batch_off[:, None] + head_off[None, :]
            tot_mask = batch_mask[:, None] & head_mask[None, :]

            blk_a = tl.load(a + tot_off, mask=tot_mask)
            blk_b = tl.load(b + tot_off, mask=tot_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
            softplus_mask = beta * x <= threshold
            softplus_x = tl.where(
                softplus_mask, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
            )
            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))

            # Store
            tl.store(g + tot_off, blk_g.to(g.dtype.element_ty), mask=tot_mask)
            tl.store(
                beta_output + tot_off,
                blk_beta_output.to(beta_output.dtype.element_ty),
                mask=tot_mask,
            )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    """
    batch, num_heads = a.shape
    seq_len = 1
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)

    core_num = get_npu_vectorcore_num()

    if batch <= core_num:
        num_progs = batch
        rows_per_prog = 1
    else:
        num_progs = core_num
        rows_per_prog = triton.cdiv(batch, core_num)

    element_size = 4  # Always cast tensor elements to f32

    def get_resident_size(block_heads):
        size_of_alog = block_heads * element_size
        size_of_bias = block_heads * element_size
        return size_of_alog + size_of_bias

    def get_dyn_size(block_heads):
        size_of_a = block_heads * element_size
        size_of_b = block_heads * element_size
        size_of_interx = block_heads * element_size * 2
        size_of_spmask = block_heads
        size_of_exp_alog = block_heads * element_size * 2
        return (
            size_of_a + size_of_b + size_of_interx + size_of_spmask + size_of_exp_alog
        )

    ub_size = 196608
    # 1. Row first
    cols_per_row = triton.cdiv(
        get_resident_size(num_heads) + get_dyn_size(num_heads), ub_size
    )
    BLOCK_HEADS = num_heads // cols_per_row
    # 2. Col next
    if rows_per_prog == 1:
        BLOCK_BATCHES = 1
    else:
        BLOCK_BATCHES = (
            (ub_size - get_resident_size(num_heads)) // get_dyn_size(num_heads) // 2
        )

    grid = (num_progs, seq_len)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        beta,
        threshold,
        rows_per_prog,
        cols_per_row,
        batch,
        num_heads,
        BLOCK_BATCHES,
        BLOCK_HEADS,
    )
    return g, beta_output
