# Copyright © 2025 Huawei Technologies Co., Ltd.

import torch
import triton
import triton.language as tl
from triton_ascend_kernels.utils import get_npu_aicore_num


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    """
    The task blocks are numbered as follows under the traditional row-wise (horizontal) partitioning:
    [0,  1,  2,  3,  4,  5,  6,  7]
    [8,  9,  10, 11, 12, 13, 14, 15]
    [16, 17, 18, 19, 20, 21, 22, 23]
    [24, 25, 26, 27, 28, 29, 30, 31]
    [32, 33, 34, 35, 36, 37, 38, 39]
    [40, 41, 42, 43, 44, 45, 46, 47]
    [48, 49, 50, 51, 52, 53, 54, 55]
    [56, 57, 58, 59, 60, 61, 62, 63]
    Core 0 handles: 0, 20, 40, 60 (4 task blocks)
    Core 1 handles: 1, 21, 41, 61 (4 task blocks)
    Core 2 handles: 2, 22, 42, 62 (4 task blocks)
    ...
    Core 19 handles: 19, 39, 59 (3 task blocks)

    For large shapes, using the traditional row-wise partitioning has the following issues:
    1. At the same time, a large number of cores need to access the same block of the left matrix memory,
       which causes bank conflicts and reduces hardware memory access efficiency.
    2. When a full row of mat_c is computed, all the data of the right matrix has already been used.
       If the right matrix is large, it can exceed the L2 cache capacity, causing frequent L2 cache loads and evictions.
       As a result, each subsequent row computation may incur cache misses, leading to low L2 cache hit rates and
       reduced operator execution efficiency.

    To address these issues, an 8 * 8 diagonal partitioning scheme can be used. Tasks are partitioned along 8 * 8 blocks diagonally,
    which significantly alleviates the problems above.

    Taking the 8 * 8 diagonal partitioning as an example, in practice the BLOCK_THRESHOLD parameter is tuned to select the optimal threshold.
    Within an 8 * 8 block, the task block numbering is as follows:
    [0,  8,  16, 24, 32, 40, 48, 56]
    [57, 1,  9,  17, 25, 33, 41, 49]
    [50, 58, 2,  10, 18, 26, 34, 42]
    [43, 51, 59, 3,  11, 19, 27, 35]
    [36, 44, 52, 60, 4,  12, 20, 28]
    [29, 37, 45, 53, 61, 5,  13, 21]
    [22, 30, 38, 46, 54, 62, 6,  14]
    [15, 23, 31, 39, 47, 55, 63, 7]

    When the M-axis exceeds 8 basic blocks, using diagonal partitioning can significantly reduce bank conflicts.
    When the right matrix size exceeds the L2 cache size, diagonal partitioning can improve L2 cache utilization.
    Therefore, enabling diagonal partitioning when both the M and N dimensions exceed 8 blocks provides performance optimization.
    The optimization is especially significant when the right matrix size exceeds the L2 cache capacity.
    """
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    # When the workload is large, the diagonal partitioning strategy can be enabled for optimization.
    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            # Implementation of 8 * 8 diagonal partitioning in code
            curThresholdM = (
                BLOCK_TRESHHOLD
                if block_idx
                < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N
                else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            )
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = (
                BLOCK_TRESHHOLD
                if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD)
                < (curThresholdM * NUM_BLOCKS_N)
                // curThresholdM_thresholdN
                * curThresholdM_thresholdN
                else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            )
            localRelativeBlock = (
                block_idx
                % (BLOCK_TRESHHOLD * NUM_BLOCKS_N)
                % (BLOCK_TRESHHOLD * curThresholdM)
            )
            task_m_idx = (
                localRelativeBlock % curThresholdM
                + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            )
            # Compute the least common multiple to easily determine the coordinates of the basic blocks.
            x, y = (
                curThresholdM,
                curThresholdN if curThresholdM > curThresholdN else curThresholdN,
                curThresholdM,
            )
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (
                localRelativeBlock + (localRelativeBlock // lcm)
            ) % curThresholdN + block_idx % (
                BLOCK_TRESHHOLD * NUM_BLOCKS_N
            ) // curThresholdM_thresholdN * BLOCK_TRESHHOLD

            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
    else:
        # traditional row-wise partitioning
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)


def matmul(
    mat_a,
    mat_b,
):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)

    """
    NPU are more friendly to 512B-aligned scenarios. The following block partitioning generally achieves good performance,
    and the optimal configuration can be selected using autotuning.
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    """

    num_cores = get_npu_aicore_num()

    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    return mat_c
