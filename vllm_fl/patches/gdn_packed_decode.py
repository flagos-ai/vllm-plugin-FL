# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# The Triton kernel below is adapted from flash-linear-attention via vLLM.
# The original source was distributed under the MIT license.
"""Numerical compatibility patch for vLLM's packed GDN decode kernel."""

from __future__ import annotations

import importlib
import inspect
import logging

from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

_TARGET_MODULE = "vllm.model_executor.layers.fla.ops.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
_VULNERABLE_BETA_EXPRESSION = (
    "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
)


@triton.jit
def _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x

    # Keep sigmoid(b) in FP32. Rounding it to the input dtype perturbs every
    # recurrent state update and compounds across decode steps.
    beta_val = tl.sigmoid(b_val)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


_fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta._fl_fp32_beta = True


def _kernel_needs_beta_patch(kernel) -> bool:
    if getattr(kernel, "_fl_fp32_beta", False):
        return False

    python_fn = getattr(kernel, "fn", kernel)
    try:
        source = inspect.getsource(python_fn)
    except (OSError, TypeError):
        # vLLM 0.24.0 is known to need the fix. If source inspection is not
        # available (for example in a stripped wheel), prefer the corrected
        # implementation over silently retaining the precision bug.
        return True
    return _VULNERABLE_BETA_EXPRESSION in source


def patch_vllm_packed_gdn_beta() -> bool:
    """Replace the vulnerable vLLM packed GDN kernel with the FP32-beta one.

    Returns ``True`` only when the replacement is applied. The symbol/source
    checks make the hook idempotent and allow it to no-op once vLLM contains an
    equivalent upstream fix.
    """
    try:
        target_module = importlib.import_module(_TARGET_MODULE)
        current_kernel = getattr(target_module, _TARGET_NAME)
    except (ImportError, AttributeError) as exc:
        logger.debug("Packed GDN decode kernel is unavailable: %s", exc)
        return False

    if not _kernel_needs_beta_patch(current_kernel):
        return False

    setattr(
        target_module,
        _TARGET_NAME,
        _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta,
    )
    logger.info("Patched vLLM packed GDN decode to keep beta in FP32")
    return True
