# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU port of ``fused_sigmoid_gating_delta_rule_update``.

Upstream launches ``grid = (NK, NV, N * HV)``; GCU caps ``grid.z`` at 255, so a
decode batch with ``N * HV > 255`` (27B: 8*48=384, 35B: 8*32=256) dies with
``OutOfResources: grid.z``.  This module keeps the exact upstream math but

* flattens the grid to 2-D (``grid.x <= 65535``, ``grid.y <= 255``);
* hoists loop-invariant scalars (A_log / dt_bias) out of the token loop;
* elides masks when ``BK == K`` and ``V % BV == 0``;
* optionally uses exp2-based math (measured; off by default because it is not
  faster on this kernel).

No upstream call site changes: ``apply_fused_sigmoid_gating_gcu_patch()``
rebinds the host function and every known ``from ... import`` binding.
"""

from __future__ import annotations

import logging

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

GCU_MAX_GRID_X = 65535
GCU_MAX_GRID_YZ = 255

_patched = False
_IMPORTERS = [
    "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating",
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
]



def _gcu_grid(total: int) -> tuple[int, int]:
    grid_x = min(total, GCU_MAX_GRID_X)
    grid_y = triton.cdiv(total, grid_x)
    if grid_y > GCU_MAX_GRID_YZ:
        grid_y = GCU_MAX_GRID_YZ
        grid_x = min(triton.cdiv(total, grid_y), GCU_MAX_GRID_X)
    return grid_x, grid_y


# ---------------------------------------------------------------------------
# General kernel (multi-token / spec decode): flattened 2-D grid.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def _kernel_gcu(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    MASK_K: tl.constexpr,
    MASK_V: tl.constexpr,
):
    # Upstream (i_k, i_v, i_nh) -> flat_pid (i_k is always 0: NK == 1).
    flat_pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    i_nh = flat_pid // NV
    if i_nh >= N * HV:
        return
    i_v = flat_pid % NV
    i_k = 0
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv

    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        if MASK_K or MASK_V:
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        else:
            b_h += tl.load(p_h0).to(tl.float32)

    for i_t in range(0, T):
        if MASK_K:
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        else:
            b_q = tl.load(p_q).to(tl.float32)
            b_k = tl.load(p_k).to(tl.float32)
        if MASK_V:
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        else:
            b_v = tl.load(p_v).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        if not IS_KDA:
            x = tl.load(p_a).to(tl.float32) + b_dt_bias
        elif MASK_K:
            x = tl.load(p_a, mask=mask_k, other=0).to(tl.float32) + b_dt_bias
        else:
            x = tl.load(p_a).to(tl.float32) + b_dt_bias
        softplus_x = tl.where(
            beta * x <= threshold, (1.0 / beta) * tl.log(1.0 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = tl.sigmoid(b_b)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.math.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.math.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        if MASK_V:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                if MASK_K or MASK_V:
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                else:
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            if MASK_K or MASK_V:
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
            else:
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))

        if not IS_KDA:
            p_a += HV
        else:
            p_a += HV * K
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV


def fused_sigmoid_gating_delta_rule_update_gcu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    # Tuned on L600: 32-element V tile, 4 warps (see PERF_REPORT).
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "NK > 1 is not supported yet"

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(1, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid_x, grid_y = _gcu_grid(NV * N * HV)
    kernel = _kernel_gcu
    common = dict(
        A_log=A_log,
        a=a if a.is_contiguous() else a.contiguous(),
        b=b if b.is_contiguous() else b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q if q.is_contiguous() else q.contiguous(),
        k=k if k.is_contiguous() else k.contiguous(),
        v=v if v.is_contiguous() else v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NV=NV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        MASK_K=BK != K,
        MASK_V=(V % BV) != 0,
        num_warps=4,
        num_stages=3,
    )
    kernel[(grid_x, grid_y)](**common)
    return o.squeeze(0), final_state


def apply_fused_sigmoid_gating_gcu_patch() -> None:
    """Rebind the upstream host function (and its importers) to the GCU one."""
    global _patched
    if _patched:
        return
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return
    try:
        import importlib

        patched_any = False
        for module_name in _IMPORTERS:
            try:
                mod = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(mod, "fused_sigmoid_gating_delta_rule_update"):
                mod.fused_sigmoid_gating_delta_rule_update = (
                    fused_sigmoid_gating_delta_rule_update_gcu
                )
                patched_any = True
        if not patched_any:
            return
        _patched = True
        logger.info(
            "Patched fused_sigmoid_gating_delta_rule_update for GCU "
            "(2-D flattened grid, grid.x <= %d, grid.y <= %d)",
            GCU_MAX_GRID_X,
            GCU_MAX_GRID_YZ,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to patch fused_sigmoid_gating_delta_rule_update for GCU: %s",
            exc,
        )
