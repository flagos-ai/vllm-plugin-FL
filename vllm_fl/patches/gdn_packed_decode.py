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

This module deliberately patches by capability rather than by a hard-coded
vLLM version.  It is safe to import on platforms without Triton, and it is a
no-op when the target module/symbol is unavailable, when an upstream kernel is
already fixed, or when the target source does not contain the vulnerable
sigmoid-cast sequence.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
from typing import Any

from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = logging.getLogger(__name__)

_TARGET_MODULE = "vllm.model_executor.layers.fla.ops.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"

# Keep this expression tied to the exact bug.  Removing whitespace before the
# comparison allows formatting changes across vLLM branches while avoiding a
# broad match that could alter a different beta/gating implementation.
_VULNERABLE_BETA_EXPRESSION = (
    "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
)
_GDN_SUBTILE_V = 16
_selection_logged = False


def _strict_patch_requested() -> bool:
    return os.environ.get("VLLM_FL_GDN_STRICT_PATCH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strict_patch_failure(reason: str, exc: BaseException | None = None) -> None:
    """Raise only when deployment explicitly requires this candidate.

    The normal plugin import path remains portable across vLLM/vendor builds,
    while the validation and production launch can set the gate to make an
    accidental fallback fail fast instead of silently using the old kernel.
    """

    if _strict_patch_requested():
        error = f"VLLM_FL_GDN_STRICT_PATCH=1: {reason}"
        if exc is None:
            raise RuntimeError(error)
        raise RuntimeError(error) from exc


def _log_selected_kernel() -> None:
    """Emit one process-level record of the actual packed-GDN selection."""

    global _selection_logged
    if _selection_logged:
        return
    _selection_logged = True
    logger.info(
        "GDN packed decode selected: kernel=%s BV=wrapper(min(nextpow2(V),32)) "
        "num_warps=wrapper(1) num_stages=wrapper(3) FP32_beta=1 "
        "FP32_state=1 register_layout=two_%d_row_subtiles",
        _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta.__name__,
        _GDN_SUBTILE_V,
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


def _kernel_source(kernel: Any) -> str | None:
    """Return source for a Triton kernel or ``None`` when unavailable."""

    python_fn = getattr(kernel, "fn", kernel)
    try:
        return inspect.getsource(python_fn)
    except (OSError, TypeError):
        return None


def _kernel_needs_beta_patch(kernel: Any) -> bool:
    """Check that ``kernel`` is the known vulnerable packed-GDN variant."""

    if getattr(kernel, "_fl_fp32_beta", False):
        return False

    source = _kernel_source(kernel)
    if source is None:
        logger.debug(
            "Cannot inspect packed GDN kernel source; preserving the vendor "
            "implementation instead of guessing its ABI"
        )
        return False

    normalized_source = re.sub(r"\s+", "", source)
    return _VULNERABLE_BETA_EXPRESSION in normalized_source


def _has_known_triton_abi(kernel: Any) -> bool:
    """Require the same Triton JIT type and Python signature as our fix."""

    if not HAS_TRITON:
        return False
    replacement = _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    if type(kernel) is not type(replacement):
        return False
    current_fn = getattr(kernel, "fn", None)
    replacement_fn = getattr(replacement, "fn", None)
    if current_fn is None or replacement_fn is None:
        return False
    try:
        current_parameters = tuple(inspect.signature(current_fn).parameters)
        replacement_parameters = tuple(
            inspect.signature(replacement_fn).parameters
        )
    except (TypeError, ValueError):
        return False
    return current_parameters == replacement_parameters


def patch_vllm_packed_gdn_beta() -> bool:
    """Replace a vulnerable packed GDN kernel with the FP32-beta variant.

    Returns ``True`` only when a replacement is applied.  Import, symbol and
    source checks make this hook idempotent and keep model/platform variants
    that do not use the legacy sigmoid-cast path untouched.
    """

    try:
        target_module = importlib.import_module(_TARGET_MODULE)
        current_kernel = getattr(target_module, _TARGET_NAME)
    except (ImportError, AttributeError) as exc:
        logger.debug("Packed GDN decode kernel is unavailable: %s", exc)
        _strict_patch_failure("packed GDN decode kernel is unavailable", exc)
        return False

    # A second plugin/model registration must not replace or obscure the
    # candidate.  Log this path too, so a model process has exactly one
    # positive selection record even when the hook is called idempotently.
    if current_kernel is _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta:
        _log_selected_kernel()
        return False

    if not _has_known_triton_abi(current_kernel):
        logger.debug(
            "Packed GDN kernel is not the known Triton ABI; preserving %r",
            current_kernel,
        )
        _strict_patch_failure("packed GDN kernel ABI is not the expected Triton ABI")
        return False
    if not _kernel_needs_beta_patch(current_kernel):
        _strict_patch_failure(
            "packed GDN kernel is not the vulnerable FP16/BF16-beta implementation"
        )
        return False

    setattr(
        target_module,
        _TARGET_NAME,
        _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta,
    )
    if (
        getattr(target_module, _TARGET_NAME, None)
        is not _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    ):
        _strict_patch_failure("packed GDN replacement did not bind to the target symbol")
        return False
    _log_selected_kernel()
    logger.info("Patched vLLM packed GDN decode to keep beta in FP32")
    return True


__all__ = [
    "patch_vllm_packed_gdn_beta",
    "_has_known_triton_abi",
    "_kernel_needs_beta_patch",
    "_fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta",
]
