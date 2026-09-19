# SPDX-License-Identifier: Apache-2.0
"""Bounded GLM5-Next KDA gates on top of vLLM 0.24's KDA core.

The two Triton kernels are the safe-gate branches from the supplied vLLM
reference implementation.  The recurrent/chunk KDA math is delegated to the
unchanged vLLM 0.24 FLA core after producing the same gate tensors.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops import kda as _kda
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv, next_power_of_2


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=num_warps)
        for bd in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["H", "D", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _safe_gate_cumsum_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(
            tl.float32
        )
        b_g += b_bias[None, :]

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))
    b_gate = lower_bound / (1.0 + tl.exp(-(b_a * b_g)))

    # Chunk-local inclusive cumsum, stored in log2 units for the downstream
    # exp2-based KDA core.
    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def fused_safe_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = _kda.FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1
    batch, tokens, heads, dim = raw_g.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _kda.prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = (
        cdiv(tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    out = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)

    def grid(meta):
        return (cdiv(meta["D"], meta["BD"]), num_chunks, batch * heads)

    _safe_gate_cumsum_kernel[grid](
        g=raw_g,
        A=A_log,
        y=out,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound,
        T=tokens,
        H=heads,
        D=dim,
        BT=chunk_size,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in _kda.BT_LIST_AUTOTUNE
        for nw in _kda.NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def _safe_gate_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT
    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=n_d < D, other=0.0).to(
            tl.float32
        )
        b_g += b_bias[None, :]
    b_y = lower_bound / (1.0 + tl.exp(-(b_a * b_g)))
    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_safe_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Compute ``lower_bound*sigmoid(exp(A_log)*(g+g_bias))`` in FP32."""
    orig_shape = g.shape[:-1]
    g = g.view(-1, g.shape[-1])
    tokens, hidden = g.shape
    heads = A_log.numel()
    assert heads * head_k_dim == hidden
    out = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (cdiv(tokens, meta["BT"]), heads)

    _safe_gate_kernel[grid](
        g,
        A_log,
        out,
        g_bias,
        lower_bound,
        tokens,
        heads,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )
    return out.view(*orig_shape, heads, head_k_dim)


def chunk_kda_with_safe_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float = -5.0,
):
    """Reference prefill path: bounded gate + chunk cumsum + vLLM KDA core."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = _kda.l2norm_fwd(q.contiguous())
        k = _kda.l2norm_fwd(k.contiguous())
    chunk_size = _kda.FLA_CHUNK_SIZE
    chunk_indices = (
        _kda.prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    gate = fused_safe_kda_gate_chunk_cumsum(
        raw_g.contiguous(),
        A_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    return _kda._chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v.contiguous(),
        g=gate,
        beta=beta.contiguous(),
        scale=scale,
        initial_state=(initial_state.contiguous() if initial_state is not None else None),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


__all__ = [
    "chunk_kda_with_safe_gate",
    "fused_safe_kda_gate",
    "fused_safe_kda_gate_chunk_cumsum",
]
