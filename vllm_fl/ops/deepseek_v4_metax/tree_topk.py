# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""Exact multi-row tree top-512 with device-side valid-length early exits."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_local_topk(
    logits_ptr,
    valid_ptr,
    cand_vals_ptr,
    cand_idx_ptr,
    stride_logits_row,
    NB_MAX: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(valid_ptr + row)
    # Early-exit (ml-static-waste fix): when the whole block lies beyond
    # valid_len, `chosen` is all-false in the original kernel (no stores), and
    # the caller pre-fills cand buffers with -inf/-1 every call, so returning
    # early is semantically identical. Removes the tl.topk on padding blocks
    # whose count scaled with static max_model_len. Grid stays static ->
    # graph-replay safe.
    if blk * BC >= valid_len:
        return
    offs = tl.arange(0, BC)
    g = blk * BC + offs
    vals = tl.load(
        logits_ptr + row * stride_logits_row + g,
        mask=g < valid_len,
        other=-float("inf"),
    )
    tv = tl.topk(vals, K)
    thresh = tl.min(tv, axis=0)
    chosen = (g < valid_len) & (vals >= thresh)
    pos = tl.cumsum(chosen.to(tl.int32), 0) - 1
    base = (row * NB_MAX + blk) * K
    tl.store(cand_vals_ptr + base + pos, vals, mask=chosen & (pos < K))
    tl.store(cand_idx_ptr + base + pos, g.to(tl.int32), mask=chosen & (pos < K))


@triton.jit
def _row_reduce_pair(
    in_vals_ptr,
    in_idx_ptr,
    out_vals_ptr,
    out_idx_ptr,
    NB_IN: tl.constexpr,
    NB_OUT: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    pid = tl.program_id(0)
    offs = tl.arange(0, BC)
    in_base = (row * NB_IN + pid * (BC // K)) * K + offs
    valid = (pid * BC + offs) < NB_IN * K
    vals = tl.load(in_vals_ptr + in_base, mask=valid, other=-float("inf"))
    idx = tl.load(in_idx_ptr + in_base, mask=valid, other=-1)
    tv = tl.topk(vals, K)
    thresh = tl.min(tv, axis=0)
    chosen = valid & (vals >= thresh)
    pos = tl.cumsum(chosen.to(tl.int32), 0) - 1
    out_base = (row * NB_OUT + pid) * K
    tl.store(out_vals_ptr + out_base + pos, vals, mask=chosen & (pos < K))
    tl.store(out_idx_ptr + out_base + pos, idx, mask=chosen & (pos < K))


def _levels(nb_max):
    """Buffer block counts per level: nb_max, ceil(/2), ..., 1."""
    levels = [nb_max]
    while levels[-1] > 1:
        levels.append(triton.cdiv(levels[-1], 2))
    return levels


def decode_topk_tree(logits, valid_lens, out_indices, k=512, block=1024):
    """Exact per-row top-k (or select-all when valid_len <= k).

    logits:      [rows, N] fp32, row-stride arbitrary; columns >= valid_lens[r]
                 are ignored (callers must ensure padded cols are not relied on).
    valid_lens:  [rows] int32/int64 on the same device.
    out_indices: [rows, k] int32; entries not selected stay -1 (fresh buffers
                 are initialized to -1 here each call).
    """
    assert logits.is_cuda and valid_lens.is_cuda and out_indices.is_cuda
    assert logits.dtype == torch.float32
    assert out_indices.shape[1] >= k
    rows = logits.shape[0]
    nb_max = triton.cdiv(logits.shape[1], block)
    levels = _levels(nb_max)
    dev = logits.device

    vals_buf = [
        torch.full((rows, lv * k), -float("inf"), dtype=torch.float32, device=dev)
        for lv in levels
    ]
    idx_buf = [
        torch.full((rows, lv * k), -1, dtype=torch.int32, device=dev) for lv in levels
    ]

    _row_local_topk[(nb_max, rows)](
        logits,
        valid_lens,
        vals_buf[0],
        idx_buf[0],
        logits.stride(0),
        NB_MAX=nb_max,
        K=k,
        BC=block,
    )
    nb = nb_max
    lvl = 1
    while nb > 1:
        nb_out = levels[lvl]
        _row_reduce_pair[(nb_out, rows)](
            vals_buf[lvl - 1],
            idx_buf[lvl - 1],
            vals_buf[lvl],
            idx_buf[lvl],
            NB_IN=nb,
            NB_OUT=nb_out,
            K=k,
            BC=2 * k,
        )
        nb = nb_out
        lvl += 1
    out_indices[:, :k] = idx_buf[-1][:, :k]
    return out_indices
