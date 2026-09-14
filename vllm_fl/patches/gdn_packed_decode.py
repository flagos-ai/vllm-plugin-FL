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
import importlib
import inspect
import logging
import re
from typing import Any

# vLLM moved the bundled flash-linear-attention package between releases:
#   0.24:  vllm.model_executor.layers.fla.ops.*
#   0.26+: vllm.third_party.flash_linear_attention.ops.*
# The upstream patch this port is based on targeted 0.24. Probe both so the
# plugin keeps working across the move instead of hard-failing at import time.
_FLA_OPS_CANDIDATES = (
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.model_executor.layers.fla.ops",
)


def _import_fla_ops(submodule: str):
    last_exc: Exception | None = None
    for _pkg in _FLA_OPS_CANDIDATES:
        try:
            return importlib.import_module(f"{_pkg}.{submodule}"), _pkg
        except ModuleNotFoundError as _e:  # noqa: PERF203
            last_exc = _e
    raise ModuleNotFoundError(
        "Could not locate vLLM's bundled flash-linear-attention ops; "
        f"tried {_FLA_OPS_CANDIDATES}"
    ) from last_exc


_fla_op_mod, _FLA_OPS_PKG = _import_fla_ops("op")
exp = _fla_op_mod.exp
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = logging.getLogger(__name__)

_TARGET_MODULE = f"{_FLA_OPS_PKG}.fused_recurrent"
_TARGET_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"

# Keep this expression tied to the exact bug.  Removing whitespace before the
# comparison allows formatting changes across vLLM branches while avoiding a
# broad match that could alter a different beta/gating implementation.
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
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid (NULL_BLOCK_ID=0).
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

    # Critical precision point: do not round sigmoid(beta) to the input
    # dtype before applying it to the FP32 recurrent state.
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


def _probe_record(outcome, detail=""):
    """Append this process's patch outcome to the sentinel log, if enabled.

    vLLM spawns EngineCore/Worker with a *filtered* environment, so a plain
    env var never reaches the process that actually runs the model.  The
    orchestration script drops /tmp/vllm_fl_gdn_probe instead and every
    process appends one line here.  A missing worker line is itself the
    finding: it means this hook never ran in that process.
    """
    try:
        import os
        if not os.path.exists("/tmp/vllm_fl_gdn_probe"):
            return
        import time
        line = "pid=%d ppid=%d t=%s outcome=%s" % (
            os.getpid(), os.getppid(), time.strftime("%H:%M:%S"), outcome)
        if detail:
            line += " detail=%s" % detail
        with open("/tmp/vllm_fl_gdn_probe.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:
        pass


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
        _probe_record("unavailable", repr(exc)[:80])
        return False

    if not _has_known_triton_abi(current_kernel):
        logger.debug(
            "Packed GDN kernel is not the known Triton ABI; preserving %r",
            current_kernel,
        )
        _probe_record("abi_mismatch", type(current_kernel).__name__)
        return False
    if not _kernel_needs_beta_patch(current_kernel):
        _probe_record("skipped_no_need",
                      "already_fp32"
                      if getattr(current_kernel, "_fl_fp32_beta", False)
                      else "source_not_vulnerable")
        return False

    setattr(
        target_module,
        _TARGET_NAME,
        _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta,
    )
    logger.info("Patched vLLM packed GDN decode to keep beta in FP32")
    _probe_record("patched", _TARGET_MODULE)
    return True


__all__ = [
    "patch_vllm_packed_gdn_beta",
    "_has_known_triton_abi",
    "_kernel_needs_beta_patch",
    "_fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta",
]
