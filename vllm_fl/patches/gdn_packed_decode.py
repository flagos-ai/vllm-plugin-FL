# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# The Triton kernel below is adapted from flash-linear-attention via vLLM.
# The original source was distributed under the MIT license.
"""Numerical compatibility patch for vLLM's packed GDN decode kernel.

The vLLM 0.24 packed Gated Delta Rule decode kernel computes ``sigmoid(beta)``
in the input dtype and only converts the result to FP32 afterwards.  With a
BF16/FP16 checkpoint this rounds every recurrent update before it is applied to
the state, which can accumulate over a long decode.  The replacement below
keeps the sigmoid in FP32, matching the prefill/chunked GDN paths.

The patch is applied directly: when the target symbol exists it is replaced
with the FP32-beta kernel, and when the source cannot be inspected the fix is
still preferred over silently keeping the rounded update.  Builds that do not
ship the FLA implementation are left untouched.
"""

from __future__ import annotations

import importlib
import inspect
import logging

from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

_TARGET_MODULE = "vllm.model_executor.layers.fla.ops.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"

# Keep this expression tied to the exact bug so the check cannot match a
# different beta/gating implementation.
_VULNERABLE_BETA_EXPRESSION = (
    "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
)
_GDN_SUBTILE_V = 16


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
    """Packed decode kernel with FP32 beta/recurrent-state arithmetic.

    The signature and launch geometry intentionally match vLLM 0.24's
    ``fused_recurrent_gated_delta_rule_packed_decode_kernel``.  Keeping this
    as a replacement for the kernel symbol (rather than changing the Python
    wrapper) also preserves vLLM's existing state/cache and CUDA-graph paths.
    """

    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)

    # Keep the invalid-state behavior observable for every output lane.  The
    # valid path below uses two 16-row subtiles when BV=32; each row is
    # independent, so this lowers the live recurrent tile from 32xBK to
    # 16xBK while keeping the wrapper's BV/grid contract unchanged.
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid (NULL_BLOCK_ID=0).
    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)

    # Normalize q and k in separate live ranges.  Loading both raw vectors
    # before the first reduction keeps two extra 128-wide FP32 values live on
    # Hopper; this ordering preserves FP32 QK normalization while reducing
    # peak register pressure for the state tile below.
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x

    # Critical precision point: do not round sigmoid(beta) to the input
    # dtype before applying it to the FP32 recurrent state.
    beta_val = tl.sigmoid(b_val)

    # Keep the tile width literal: Triton 3.6 treats Python module globals
    # differently across cache/codegen paths, while this value is part of the
    # deliberate register-layout choice.  The second mask also preserves the
    # original ABI if a caller ever supplies BV < 16 explicitly.
    for v_start in range(0, BV, 16):
        v_lane = tl.arange(0, 16)
        o_v_block = i_v * BV + v_start + v_lane
        mask_v_block = (v_start + v_lane < BV) & (o_v_block < V)
        mask_h_block = mask_v_block[:, None] & mask_k[None, :]
        p_o_block = o + (i_n * HV + i_hv) * V + o_v_block
        p_h0_block = p_h0 + o_v_block[:, None] * K + o_k[None, :]
        p_ht_block = ht + state_idx * stride_final_state_token
        p_ht_block = (
            p_ht_block
            + i_hv * V * K
            + o_v_block[:, None] * K
            + o_k[None, :]
        )
        v_off_block = (2 * H * K) + i_hv * V + o_v_block
        b_v_block = tl.load(
            p_mixed + v_off_block, mask=mask_v_block, other=0
        ).to(tl.float32)
        b_h_block = tl.load(
            p_h0_block, mask=mask_h_block, other=0
        ).to(tl.float32)
        b_h_block *= exp(g_val)
        b_v_block -= tl.sum(b_h_block * b_k[None, :], 1)
        b_v_block *= beta_val
        b_h_block += b_v_block[:, None] * b_k[None, :]
        b_o_block = tl.sum(b_h_block * b_q[None, :], 1)
        tl.store(
            p_o_block,
            b_o_block.to(p_o_block.dtype.element_ty),
            mask=mask_v_block,
        )
        tl.store(
            p_ht_block,
            b_h_block.to(p_ht_block.dtype.element_ty),
            mask=mask_h_block,
        )


# Marker used for idempotence.  Triton JIT functions accept Python attributes
# in the supported vLLM/Triton versions; the marker is also harmless for the
# CPU-side Triton placeholder used during import-only tests.
_fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta._fl_fp32_beta = True


def _kernel_needs_beta_patch(kernel) -> bool:
    """Check that ``kernel`` is the known vulnerable packed-GDN variant."""

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


__all__ = [
    "patch_vllm_packed_gdn_beta",
    "_kernel_needs_beta_patch",
    "_fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta",
]
