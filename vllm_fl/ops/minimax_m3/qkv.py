# SPDX-License-Identifier: Apache-2.0
"""M3 BF16 QK Gemma norm, partial NeoX RoPE and exact paged cache insertion."""

import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _norm_rope(
    X,
    Y,
    W,
    CS,
    POS,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    CSS: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.arange(0, 128)
    x = tl.load(X + m * XS0 + h * XS1 + c).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    y = (
        (x * tl.rsqrt(tl.sum(x * x, 0) / 128 + EPS) * (1.0 + w))
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    half = RD // 2
    idx = tl.where(c < half, c + half, c - half)
    pair = tl.gather(y, idx, 0)
    pos = tl.load(POS + m)
    cos = tl.load(CS + pos * CSS + c % half, c < RD, other=1.0).to(tl.float32)
    sin = tl.load(CS + pos * CSS + half + c % half, c < RD, other=0.0).to(tl.float32)
    rot = y * cos + tl.where(c < half, -pair, pair) * sin
    tl.store(Y + m * YS0 + h * YS1 + c, tl.where(c < RD, rot, y))


@libentry()
@triton.jit
def _put_kv(
    K,
    V,
    C,
    S,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    BS: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    which = tl.program_id(2)
    c = tl.arange(0, 128)
    slot = tl.load(S + m)
    p = tl.where(which == 0, K + m * KS0 + h * KS1 + c, V + m * VS0 + h * VS1 + c)
    val = tl.load(p)
    ptr = C + (slot // BS) * C0 + which * C1 + (slot % BS) * C2 + h * C3 + c * C4
    tl.store(ptr, val, slot >= 0)


@libentry()
@triton.jit
def _put_index(
    K,
    C,
    S,
    KS: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BS: tl.constexpr,
):
    m = tl.program_id(0)
    c = tl.arange(0, 128)
    slot = tl.load(S + m)
    val = tl.load(K + m * KS + c)
    tl.store(C + (slot // BS) * C0 + (slot % BS) * C1 + c * C2, val, slot >= 0)


def qknorm_rope_insert(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight=None,
    index_k_norm_weight=None,
    num_index_heads=0,
    slot_mapping=None,
    index_slot_mapping=None,
    kv_cache=None,
    index_cache=None,
    block_size=0,
    q_out=None,
    index_q_out=None,
    kv_cache_dtype="auto",
):
    assert (
        qkv.is_cuda
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 2
        and qkv.stride(-1) == 1
    )
    assert rotary_dim == 64 and kv_cache_dtype in ("auto", "bfloat16")
    m = qkv.shape[0]
    nq = num_heads * 128
    nk = num_kv_heads * 128
    ni = num_index_heads * 128
    assert qkv.shape[1] == nq + 2 * nk + (ni + 128 if num_index_heads else 0)
    q = qkv[:, :nq].view(m, num_heads, 128)
    k = qkv[:, nq : nq + nk].view(m, num_kv_heads, 128)
    v = qkv[:, nq + nk : nq + 2 * nk].view(m, num_kv_heads, 128)

    def norm(x, w, out=None):
        y = x if out is None else out.view(x.shape)
        assert y.dtype == x.dtype and w.shape == (128,) and y.stride(-1) == 1
        if m:
            _norm_rope[(m, x.shape[1])](
                x,
                y,
                w,
                cos_sin_cache,
                positions,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                cos_sin_cache.stride(0),
                rotary_dim,
                eps,
                num_warps=4,
                enable_fp_fusion=False,
            )

    norm(q, q_norm_weight, q_out)
    norm(k, k_norm_weight)
    if num_index_heads:
        iq = qkv[:, nq + 2 * nk : nq + 2 * nk + ni].view(m, num_index_heads, 128)
        ik = qkv[:, -128:].view(m, 1, 128)
        norm(iq, index_q_norm_weight, index_q_out)
        norm(ik, index_k_norm_weight)
    if kv_cache is not None:
        assert (
            kv_cache.dtype == qkv.dtype
            and kv_cache.ndim == 5
            and slot_mapping is not None
        )
        assert block_size == kv_cache.shape[2] and kv_cache.shape[3:] == (
            num_kv_heads,
            128,
        )
        if m:
            _put_kv[(m, num_kv_heads, 2)](
                k,
                v,
                kv_cache,
                slot_mapping,
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                *kv_cache.stride(),
                block_size,
                num_warps=4,
            )
    if index_cache is not None:
        assert (
            num_index_heads and index_cache.dtype == qkv.dtype and index_cache.ndim == 3
        )
        slots = index_slot_mapping if index_slot_mapping is not None else slot_mapping
        assert slots is not None
        if m:
            _put_index[(m,)](
                ik,
                index_cache,
                slots,
                ik.stride(0),
                *index_cache.stride(),
                index_cache.shape[1],
                num_warps=4,
            )


def register(registry):
    from .torch_ops import qknorm_rope_insert as namespace_entry
    from vllm_fl.dispatch import BackendImplKind, OpImpl

    registry.register_impl(
        OpImpl(
            op_name="m3_qknorm_rope_insert",
            impl_id="m3.flagos.qknorm_rope_insert",
            kind=BackendImplKind.DEFAULT,
            fn=namespace_entry,
            priority=150,
        )
    )
