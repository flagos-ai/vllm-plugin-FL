# SPDX-License-Identifier: Apache-2.0
"""M3 extensions registered with FL. Original implementations stay available in patch fallbacks."""

import torch
import triton
import triton.language as tl
from flag_gems.ops.gemma_rms_norm import gemma_rms_norm_kernel
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _gemma_add(
    X,
    R,
    W,
    N: tl.constexpr,
    XS: tl.constexpr,
    RS: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    v = tl.load(X + row * XS + c, c < N, 0).to(tl.float32) + tl.load(
        R + row * RS + c, c < N, 0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, 0) / N + EPS)
    w = tl.load(W + c, c < N, 0).to(tl.float32)
    tl.store(R + row * RS + c, v, c < N)
    tl.store(X + row * XS + c, v * inv * (1.0 + w), c < N)


@libentry()
@triton.jit
def _swiglu(
    X,
    Y,
    N: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    TOTAL: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // N
    c = i % N
    g = tl.minimum(tl.load(X + row * XS + c, i < TOTAL, 0).to(tl.float32), LIMIT)
    u = tl.minimum(
        tl.maximum(tl.load(X + row * XS + N + c, i < TOTAL, 0).to(tl.float32), -LIMIT),
        LIMIT,
    )
    y = g / (1.0 + tl.exp(-ALPHA * g)) * (u + BETA)
    tl.store(Y + row * YS + c, y, i < TOTAL)


def _matrix(x):
    if x.ndim == 2 and x.stride(-1) == 1:
        return x
    if x.is_contiguous():
        return x.view(-1, x.shape[-1])
    raise ValueError(
        "M3 extension requires contiguous last dimension and viewable rows"
    )


def gemma(x, weight, eps, residual=None):
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("M3 Gemma requires a supported GPU tensor")
    a = _matrix(x)
    n = a.shape[-1]
    if weight.shape != (n,) or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("Invalid Gemma weight")
    if residual is None:
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        b = y.view(-1, n)
        inv = torch.empty(a.shape[0], device=x.device, dtype=torch.float32)
        if a.numel():
            gemma_rms_norm_kernel[(a.shape[0],)](
                b,
                inv,
                a,
                weight,
                b.stride(0),
                1,
                a.stride(0),
                1,
                n,
                eps,
                triton.next_power_of_2(n),
            )
        return y
    r = _matrix(residual)
    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("Invalid Gemma residual")
    if x.data_ptr() == residual.data_ptr():
        raise ValueError("Gemma input/residual must not alias")
    if a.numel():
        _gemma_add[(a.shape[0],)](
            a,
            r,
            weight,
            n,
            a.stride(0),
            r.stride(0),
            eps,
            triton.next_power_of_2(n),
            num_warps=4,
        )
    return x, residual


def swiglu(x, limit=7.0, alpha=1.702, beta=1.0, out=None):
    a = _matrix(x)
    if not x.is_cuda or a.shape[-1] % 2:
        raise ValueError("SwiGLU requires GPU gate/up halves")
    n = a.shape[-1] // 2
    if out is None:
        out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if (
        out.shape != (*x.shape[:-1], n)
        or out.dtype != x.dtype
        or out.device != x.device
    ):
        raise ValueError("Invalid SwiGLU output")
    y = _matrix(out)
    if y.numel():
        _swiglu[(triton.cdiv(y.numel(), 256),)](
            a, y, n, a.stride(0), y.stride(0), y.numel(), alpha, beta, limit, 256
        )
    return out


def index_score(*a, **kw):
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_score

    return minimax_m3_index_score(*a, **kw)


def index_topk(*a, **kw):
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_topk

    return minimax_m3_index_topk(*a, **kw)


def index_decode(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out=None,
):
    if decode_query_len == 1:
        # Exact eager fallback: identical full block scores, causal range and top-k budget.
        # The native split top-k merge currently stalls in the MetaX Triton compiler.
        cu = torch.arange(seq_lens.shape[0] + 1, device=idx_q.device, dtype=torch.int32)
        prefix = seq_lens - 1
        score = index_score(
            idx_q,
            index_kv_cache,
            block_table,
            cu,
            seq_lens,
            prefix,
            1,
            max_seq_len,
            num_kv_heads,
        )
        return index_topk(
            score, cu, prefix, 1, topk, init_blocks, local_blocks, out=out
        )
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_decode

    return minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out=out,
    )


def sparse_attn(*a, **kw):
    from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn

    return minimax_m3_sparse_attn(*a, **kw)


def sparse_decode(*a, **kw):
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn_decode,
    )

    return minimax_m3_sparse_attn_decode(*a, **kw)


def register(registry):
    from . import torch_ops
    from .vision import register as register_vision

    register_vision(registry)
    from .qkv import register

    register(registry)
    from vllm_fl.dispatch import BackendImplKind, OpImpl

    for name, fn in {
        "gemma": gemma,
        "swiglu": swiglu,
        "index_score": index_score,
        "index_topk": index_topk,
        "index_decode": index_decode,
        "sparse_attn": sparse_attn,
        "sparse_decode": sparse_decode,
    }.items():
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="m3.flagos." + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(torch_ops, name),
                priority=150,
            )
        )
    registry.register_impl(
        OpImpl(
            op_name="m3_patch_embed",
            impl_id="m3.flagos.patch_embed",
            kind=BackendImplKind.DEFAULT,
            fn=torch_ops.patch_embed,
            priority=150,
        )
    )
