# Copyright (c) 2026 BAAI. All rights reserved.

"""Guarded DV4 decode routing, reusing FlagGems' top-k ordering network."""

import os

import torch
import torch.nn.functional as F
import flag_gems
import triton
import triton.language as tl
from flag_gems.ops.topk import argsort


@triton.jit
def _sum_six(weights, slots, GEMS_SUM: tl.constexpr):
    # The production Hygon blacklist retains Torch sum; an all-FlagGems
    # context uses a different tree. Preserve the active six-value order.
    # One FP32 ulp here can cross downstream LightOp BF16 rounding boundaries.
    w0 = tl.sum(tl.where(slots == 0, weights, 0.0), 0)
    w1 = tl.sum(tl.where(slots == 1, weights, 0.0), 0)
    w2 = tl.sum(tl.where(slots == 2, weights, 0.0), 0)
    w3 = tl.sum(tl.where(slots == 3, weights, 0.0), 0)
    w4 = tl.sum(tl.where(slots == 4, weights, 0.0), 0)
    w5 = tl.sum(tl.where(slots == 5, weights, 0.0), 0)
    if GEMS_SUM:
        return ((w0 + w4) + w2) + ((w1 + w5) + w3)
    return ((w0 + w4) + (w1 + w5)) + (w2 + w3)


@triton.jit
def _router_bias_from_scores(X, Bias, W, I, HAS_BIAS: tl.constexpr,
                       RENORM: tl.constexpr, SCALE: tl.constexpr,
                       GEMS_SUM: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 256)
    raw = tl.load(X + row * 256 + col)
    score = raw
    if HAS_BIAS:
        score = score + tl.load(Bias + col)
    # Match the actual FlagGems topk_single_stage_kernel for equal scores;
    # selecting the smallest tied index is not equivalent for every pattern.
    _, ids = argsort(score, col, dim=0, descending=True)
    weights = tl.gather(raw, ids, axis=0)
    weights = tl.where(col < 6, weights, 0.0)
    if RENORM:
        weights = weights / tl.maximum(_sum_six(weights, col, GEMS_SUM), 1.0e-20)
    tl.store(W + row * 6 + col, weights * SCALE, col < 6)
    tl.store(I + row * 6 + col, ids, col < 6)


@triton.jit
def _router_hash_from_scores(X, Tokens, Table, W, I, TABLE_ROWS: tl.constexpr,
                       RENORM: tl.constexpr, SCALE: tl.constexpr,
                       GEMS_SUM: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 8)
    token = tl.load(Tokens + row).to(tl.int64)
    token = tl.where(token < 0, token + TABLE_ROWS, token)
    ids = tl.load(Table + token * 6 + k, k < 6, 0).to(tl.int64)
    x = tl.load(X + row * 256 + ids, k < 6, 0)
    weights = tl.where(k < 6, x, 0.0)
    if RENORM:
        weights = weights / tl.maximum(_sum_six(weights, k, GEMS_SUM), 1.0e-20)
    tl.store(W + row * 6 + k, weights * SCALE, k < 6)
    tl.store(I + row * 6 + k, ids, k < 6)


def try_router_fusion(weights, indices, logits, renormalize, scale,
                      bias=None, input_ids=None, table=None) -> bool:
    """Fuse only the validated FP32, E256/topk6 decode ABI."""
    if (
        os.getenv("VLLM_FL_HYGON_ROUTER_FUSION", "1").lower() in {"0", "off", "native"}
        or torch.version.hip is None
        or torch.__version__.split("+")[0] != "2.10.0"
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= 256
        or logits.shape[1] != 256
        or logits.requires_grad
        or weights.shape != (logits.shape[0], 6)
        or indices.shape != weights.shape
        or weights.dtype != torch.float32
        or weights.requires_grad
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        return False
    # Only replace the sorting operation when the installed FlagGems topk
    # is active. Plain PyTorch has a different unspecified tie order.
    registrar = getattr(flag_gems, "current_work_registrar", None)
    if table is None:
        if registrar is None or "topk" not in registrar.get_all_ops():
            return False
    tensors = [logits, weights, indices]
    if bias is not None:
        if bias.shape != (256,) or bias.dtype != torch.float32:
            return False
        tensors.append(bias)
    if table is not None:
        if input_ids is None:
            return False
        if (input_ids.shape != (logits.shape[0],)
                or input_ids.dtype not in (torch.int32, torch.int64)
                or table.ndim != 2 or table.shape[1] != 6
                or table.dtype not in (torch.int32, torch.int64)):
            return False
        tensors.extend([input_ids, table])
    if any(t.device != logits.device or not t.is_contiguous() for t in tensors):
        return False
    # The unfused path first materializes scores, so aliased output storage
    # must retain that path instead of overwriting inputs during this kernel.
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if len(set(storage)) != len(storage):
        return False
    # Preserve both dispatch and FP32 intermediates of the current score path.
    # Substituting log1p/exp here changed near-tied expert choices in validation.
    scores = torch.sqrt(F.softplus(logits.float()))
    gems_sum = registrar is not None and "sum.dim_IntList" in registrar.get_all_keys()
    if table is None:
        _router_bias_from_scores[(logits.shape[0],)](
            scores, bias, weights, indices, bias is not None, renormalize, scale,
            gems_sum,
            num_warps=1, enable_fp_fusion=False,
        )
    else:
        _router_hash_from_scores[(logits.shape[0],)](
            scores, input_ids, table, weights, indices, table.shape[0], renormalize,
            scale, gems_sum, num_warps=1, enable_fp_fusion=False,
        )
    return True
