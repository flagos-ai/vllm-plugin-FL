# Copyright (c) 2026 BAAI. All rights reserved.
"""Wave64 mHC mix/Sinkhorn + BF16 RMSNorm; hidden4096, hc_mult4 only."""
import math

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def _mhc_pre_norm_wave(
    hidden_size: int,
    norm_eps: float,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_pre_norm_wave(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        norm_weight: T.Tensor[(hidden_size,), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        threads = 128
        
        n_splits_aligned = tilelang.math.next_power_of_2(n_splits)
        
        if n_splits >= 4:
            split_groups = threads // 32
            # assert n_splits % split_groups == 0
            group_rows = n_splits // split_groups
        with T.Kernel(num_tokens, threads=threads) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            tx = T.get_thread_binding()
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            rms = T.alloc_fragment(1, T.float32)

            if n_splits >= 4 and n_splits % split_groups == 0:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes_pre = T.alloc_fragment((split_groups, 32), T.float32)
                mixes_aligned = T.alloc_fragment(32, T.float32)
                T.clear(mixes_pre)
                for r in T.serial(group_rows):
                    for i, j in T.Parallel(split_groups, 32):
                        if j < mhc_mult3:
                            mixes_pre[i, j] += gemm_out_mul[i * group_rows + r, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(32):
                    if i < mhc_mult3:
                        mixes_shared[i] = mixes_aligned[i] * rms[0]
            elif n_splits >= 2:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)
            else:
                rms[0] = gemm_out_sqrsum[0, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = gemm_out_mul[0, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if tx < 64:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2] + mhc_base[j * mhc_mult + k + mhc_mult * 2]

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_fragment(mhc_mult, T.float32)
                for j in T.serial(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                ###################################################################
                # _mhc_pre_apply_mix_fwd — flat local + hand-pipeline prefetch (T.serial)
                #
                #   xl[2 * tile_elems]: slot ping-pong via offset only
                #   prologue : load tile[0] -> slot0
                #   steady   : T.serial(h_iters - 1) prefetch[i+1] + compute[i]
                #   epilogue : compute last tile (no prefetch branch)
                apply_threads = 64
                vec_size = hidden_block // apply_threads
                tile_elems = mhc_mult * vec_size
                lane = tx - 64

                h_iters = hidden_size // hidden_block
                mixed_local = T.alloc_local((hidden_size // 64,), T.bfloat16)
                xl = T.alloc_local((2 * tile_elems,), T.bfloat16)
                ol = T.alloc_local((vec_size,), T.float32)

                # prologue: tile 0 -> slot 0
                for i_mhc in T.serial(mhc_mult):
                    off = i_mhc * vec_size
                    for v in T.vectorized(vec_size):
                        xl[off + v] = residual[pid, i_mhc, lane * vec_size + v]

                for i0_h in T.serial(h_iters - 1):
                    cur_slot = (i0_h % 2) * tile_elems
                    nxt_slot = ((i0_h + 1) % 2) * tile_elems
                    h_base = i0_h * hidden_block + lane * vec_size
                    next_base = (i0_h + 1) * hidden_block + lane * vec_size

                    for i_mhc in T.serial(mhc_mult):
                        off = nxt_slot + i_mhc * vec_size
                        for v in T.vectorized(vec_size):
                            xl[off + v] = residual[pid, i_mhc, next_base + v]

                    for v in T.vectorized(vec_size):
                        ol[v] = 0.0
                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        off = cur_slot + i_mhc * vec_size
                        for v in T.vectorized(vec_size):
                            ol[v] += pre * T.Cast(T.float32, xl[off + v])
                    for v in T.vectorized(vec_size):
                        mixed_local[i0_h * vec_size + v] = ol[v]

                # epilogue: last tile
                last_h = h_iters - 1
                last_slot = (last_h % 2) * tile_elems
                last_base = last_h * hidden_block + lane * vec_size
                for v in T.vectorized(vec_size):
                    ol[v] = 0.0
                for i_mhc in T.serial(mhc_mult):
                    pre = pre_mix_shared[i_mhc]
                    off = last_slot + i_mhc * vec_size
                    for v in T.vectorized(vec_size):
                        ol[v] += pre * T.Cast(T.float32, xl[off + v])
                for v in T.vectorized(vec_size):
                    mixed_local[last_h * vec_size + v] = ol[v]

                sums = T.alloc_local((4,), T.float32)
                for g in T.unroll(4):
                    sums[g] = 0.0
                    for j in T.unroll(16):
                        z = T.Cast(T.float32, mixed_local[(g + (j // 8) * 4) * 8 + j % 8])
                        sums[g] += z * z
                    sums[g] += T.shfl_xor(sums[g], 8, 64)
                    sums[g] += T.shfl_xor(sums[g], 4, 64)
                    sums[g] += T.shfl_xor(sums[g], 2, 64)
                    sums[g] += T.shfl_xor(sums[g], 1, 64)
                    sums[g] += T.shfl_xor(sums[g], 16, 64)
                    sums[g] += T.shfl_xor(sums[g], 32, 64)
                inv = T.alloc_local((1,), T.float32)
                inv[0] = 1.0 / T.sqrt(((sums[0] + sums[2]) + (sums[1] + sums[3])) / hidden_size + norm_eps)
                for h in T.unroll(hidden_size // hidden_block):
                    for v in T.vectorized(vec_size):
                        normalized = T.Cast(T.bfloat16, T.Cast(T.float32, mixed_local[h * vec_size + v]) * inv[0])
                        layer_input[pid, h * hidden_block + lane * vec_size + v] = T.Cast(T.float32, normalized) * T.Cast(T.float32, norm_weight[h * hidden_block + lane * vec_size + v])

    return mhc_pre_norm_wave
