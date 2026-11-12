# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
"""K3 compatibility differences from the pinned vLLM KDA implementation.

Upstream computation is reused through local launch bindings. Host wrappers
remain local where they select those bindings or the retained gate cumsum.
"""

import torch

from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk as native_chunk
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.op import log
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.chunk_intra import chunk_kda_fwd_intra
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.launch import local_kernel

recompute_w_u_fwd_kernel = local_kernel(
    native_chunk.recompute_w_u_fwd_kernel,
    heuristics={
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    },
    configs=[
        triton.Config(
            {"DEFAULT_DOT_PRECISION": "ieee"},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)

chunk_gla_fwd_kernel_o = local_kernel(
    native_chunk.chunk_gla_fwd_kernel_o,
    heuristics={"IS_VARLEN": lambda args: args["cu_seqlens"] is not None},
    configs=[
        triton.Config(
            {"BK": BK, "BV": BV, "DEFAULT_DOT_PRECISION": "ieee"},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for BK in [32, 64]
        for BV in [16]
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=["BT"],
)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 32
    BV = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION="ieee",
    )
    return w, u, None, kg


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * H)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in [32, 64]
        for num_warps in [4]
    ],
    key=["H", "S", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    raw_beta,
    A_log,
    g_bias,
    o,
    beta_out,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound,
    beta,
    threshold,
    T,
    stride_beta_batch,
    stride_beta_token,
    stride_beta_head,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    if i_s == 0:
        o_beta_t = tl.arange(0, BT)
        m_beta = i_t * BT + o_beta_t < T
        if IS_VARLEN:
            p_beta = (
                raw_beta
                + (bos + i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        else:
            p_beta = (
                raw_beta
                + i_b * stride_beta_batch
                + (i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        b_beta = tl.load(p_beta, mask=m_beta, other=0.0).to(tl.float32)
        p_beta_out = beta_out + (bos + i_t * BT + o_beta_t) * H + i_h
        tl.store(p_beta_out, tl.sigmoid(b_beta), mask=m_beta)
        return

    i_s -= 1

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_bias = tl.make_block_ptr(
            g_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_bias, boundary_check=(0,)).to(tl.float32)
        b_s += b_bias[None, :]

    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_s)
    else:
        b_g_scaled = b_s * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_s,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = -b_a * b_softplus

    # Boundary loads return zero, but bias and gate activation can make padded
    # rows nonzero. Padding trails valid rows, so it only affects masked stores.
    b_o = tl.cumsum(b_gate.to(tl.float64), axis=0) * tl.full(
        (), 1.4426950408889634, tl.float64
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    B, T, H, D = raw_g.shape
    if raw_beta.shape != (B, T, H):
        raise ValueError(f"Expected raw_beta shape {(B, T, H)}, got {raw_beta.shape}")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    beta_out = torch.empty(raw_beta.shape, device=raw_beta.device, dtype=torch.float32)

    def grid(meta):
        # For each (chunk, head), program 0 computes beta without extending a
        # gate tile's critical path. The remaining programs cover the gate dim.
        return (cdiv(meta["S"], meta["BS"]) + 1, NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        o=y,
        beta_out=beta_out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g). Keep this in sync with the
        # `use_exp2=True` path in `_chunk_kda_fwd_with_cumulative_g`.
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound or 0.0,
        beta=beta,
        threshold=threshold,
        T=T,
        stride_beta_batch=raw_beta.stride(0),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        H=H,
        S=D,
        BT=chunk_size,
        USE_LOWER_BOUND=lower_bound is not None,
    )
    return y, beta_out


def _chunk_kda_fwd_with_cumulative_g(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
):
    # `g` must already be chunk-local cumulatively-summed AND scaled by
    # RCP_LN2 (so the downstream exp2-based kernels reproduce exp(g)).
    # Use `chunk_kda_fwd` or `chunk_kda_with_fused_gate_fwd` instead of
    # calling this helper directly unless that invariant is upheld.
    Aqk, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    del A
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    del w, u, kg
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    del Aqk, v_new, h
    return o, final_state


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=lower_bound is not None,
    )
