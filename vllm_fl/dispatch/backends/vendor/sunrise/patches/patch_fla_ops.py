# Copyright (c) 2026 BAAI. All rights reserved.

"""Redirect FLA GDN kernels to PTPU sgl_kernel implementations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_ORIG_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE = None
_ORIG_FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE = None
_ORIG_SOLVE_TRIL = None
_ORIG_CHUNK_FWD_O = None
_ORIG_CHUNK_GATED_DELTA_RULE_FWD_H = None
_ORIG_CHUNK_SCALED_DOT_KKT_FWD = None
_ORIG_CHUNK_LOCAL_CUMSUM = None


def get_orig_fused_sigmoid_gating_delta_rule_update():
    """Return the un-patched FLA Triton ``fused_sigmoid_gating_delta_rule_update``."""
    return _ORIG_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE


def get_orig_fused_recurrent_gated_delta_rule_packed_decode():
    """Return the un-patched FLA Triton ``...packed_decode``."""
    return _ORIG_FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE


def get_orig_solve_tril():
    """Return the un-patched FLA Triton ``solve_tril``."""
    return _ORIG_SOLVE_TRIL


def get_orig_chunk_fwd_o():
    """Return the un-patched FLA Triton ``chunk_fwd_o``."""
    return _ORIG_CHUNK_FWD_O


def get_orig_chunk_gated_delta_rule_fwd_h():
    """Return the un-patched FLA Triton ``chunk_gated_delta_rule_fwd_h``."""
    return _ORIG_CHUNK_GATED_DELTA_RULE_FWD_H


def get_orig_chunk_scaled_dot_kkt_fwd():
    """Return the un-patched FLA Triton ``chunk_scaled_dot_kkt_fwd``."""
    return _ORIG_CHUNK_SCALED_DOT_KKT_FWD


def get_orig_chunk_local_cumsum():
    """Return the un-patched FLA Triton ``chunk_local_cumsum``."""
    return _ORIG_CHUNK_LOCAL_CUMSUM


def apply_patch() -> bool:
    """Idempotently install the PTPU FLA dispatch patch (all stages → sgl)."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    try:
        import torch_ptpu.sgl_kernel as _ptpu_sgl  # noqa: F401
    except Exception as exc:  # pragma: no cover - non-PTPU env
        logger.debug(
            "Skipping PTPU FLA patch (torch_ptpu.sgl_kernel not importable): %s",
            exc,
        )
        return False

    try:
        from vllm.model_executor.layers.fla.ops import (
            chunk as _fla_chunk,
            chunk_delta_h as _fla_chunk_delta_h,
            chunk_o as _fla_chunk_o,
            chunk_scaled_dot_kkt as _fla_chunk_scaled_dot_kkt,
            cumsum as _fla_cumsum,
            fused_recurrent as _fla_fused_recurrent,
            fused_sigmoid_gating as _fla_fused_sigmoid_gating,
            l2norm as _fla_l2norm,
            solve_tril as _fla_solve_tril,
            wy_fast as _fla_wy_fast,
        )
        from vllm.model_executor.layers.fla import ops as _fla_ops_pkg
    except Exception as exc:  # pragma: no cover - vLLM without GDN support
        logger.debug(
            "Skipping PTPU FLA patch (vLLM FLA modules not importable): %s",
            exc,
        )
        return False

    # Snapshot prefill-stage originals *before* rebinding. Fallback paths
    # must call these references; re-importing the same module after the
    # patch would resolve back to the PTPU wrapper and recurse.
    global _ORIG_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE
    _ORIG_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE = (
        _fla_fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update
    )
    global _ORIG_FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE
    _ORIG_FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE = (
        _fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
    )
    global _ORIG_SOLVE_TRIL
    _ORIG_SOLVE_TRIL = _fla_solve_tril.solve_tril
    global _ORIG_CHUNK_FWD_O
    _ORIG_CHUNK_FWD_O = _fla_chunk_o.chunk_fwd_o
    global _ORIG_CHUNK_GATED_DELTA_RULE_FWD_H
    _ORIG_CHUNK_GATED_DELTA_RULE_FWD_H = (
        _fla_chunk_delta_h.chunk_gated_delta_rule_fwd_h
    )
    global _ORIG_CHUNK_SCALED_DOT_KKT_FWD
    _ORIG_CHUNK_SCALED_DOT_KKT_FWD = (
        _fla_chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd
    )
    global _ORIG_CHUNK_LOCAL_CUMSUM
    _ORIG_CHUNK_LOCAL_CUMSUM = _fla_cumsum.chunk_local_cumsum

    from ..impl.fla.chunk_fwd_o import chunk_fwd_o as _ptpu_chunk_fwd_o
    from ..impl.fla.chunk_h import (
        chunk_gated_delta_rule_fwd_h as _ptpu_chunk_gated_delta_rule_fwd_h,
    )
    from ..impl.fla.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd as _ptpu_chunk_scaled_dot_kkt_fwd,
    )
    from ..impl.fla.cumsum import chunk_local_cumsum as _ptpu_chunk_local_cumsum
    from ..impl.fla.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode as _ptpu_packed_decode,
    )
    from ..impl.fla.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as _ptpu_fused_sigmoid_gating,
    )
    from ..impl.fla.l2norm import l2norm_fwd as _ptpu_l2norm_fwd
    from ..impl.fla.solve_tril import solve_tril as _ptpu_solve_tril
    from ..impl.fla.wy_fast import recompute_w_u_fwd as _ptpu_recompute_w_u_fwd

    # Prefill stages
    _fla_cumsum.chunk_local_cumsum = _ptpu_chunk_local_cumsum
    _fla_chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd = (
        _ptpu_chunk_scaled_dot_kkt_fwd
    )
    _fla_solve_tril.solve_tril = _ptpu_solve_tril
    _fla_wy_fast.recompute_w_u_fwd = _ptpu_recompute_w_u_fwd
    _fla_chunk_delta_h.chunk_gated_delta_rule_fwd_h = (
        _ptpu_chunk_gated_delta_rule_fwd_h
    )
    _fla_chunk_o.chunk_fwd_o = _ptpu_chunk_fwd_o

    _fla_chunk.chunk_local_cumsum = _ptpu_chunk_local_cumsum
    _fla_chunk.chunk_scaled_dot_kkt_fwd = _ptpu_chunk_scaled_dot_kkt_fwd
    _fla_chunk.solve_tril = _ptpu_solve_tril
    _fla_chunk.recompute_w_u_fwd = _ptpu_recompute_w_u_fwd
    _fla_chunk.chunk_gated_delta_rule_fwd_h = _ptpu_chunk_gated_delta_rule_fwd_h
    _fla_chunk.chunk_fwd_o = _ptpu_chunk_fwd_o
    _fla_chunk.l2norm_fwd = _ptpu_l2norm_fwd

    # Decode / aux
    _fla_l2norm.l2norm_fwd = _ptpu_l2norm_fwd
    _fla_fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update = (
        _ptpu_fused_sigmoid_gating
    )
    _fla_ops_pkg.fused_sigmoid_gating_delta_rule_update = (
        _ptpu_fused_sigmoid_gating
    )
    _fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        _ptpu_packed_decode
    )
    _fla_ops_pkg.fused_recurrent_gated_delta_rule_packed_decode = (
        _ptpu_packed_decode
    )

    try:
        from vllm.model_executor.layers.mamba import (
            gdn_linear_attn as _gdn_lib,
        )
    except Exception as exc:  # pragma: no cover - vLLM without GDN
        logger.debug(
            "PTPU FLA patch: gdn_linear_attn not importable, skipping "
            "module-level rebind: %s",
            exc,
        )
    else:
        _gdn_lib.fused_sigmoid_gating_delta_rule_update = (
            _ptpu_fused_sigmoid_gating
        )
        _gdn_lib.l2norm_fwd = _ptpu_l2norm_fwd
        _gdn_lib.fused_recurrent_gated_delta_rule_packed_decode = (
            _ptpu_packed_decode
        )

    _PATCH_APPLIED = True
    logger.info(
        "Applied PTPU FLA patch: all 6 GDN prefill stages + spec/mixed "
        "decode + l2norm + packed_decode route to torch_ptpu.sgl_kernel."
    )
    return True


__all__ = [
    "apply_patch",
    "get_orig_chunk_fwd_o",
    "get_orig_chunk_gated_delta_rule_fwd_h",
    "get_orig_chunk_local_cumsum",
    "get_orig_chunk_scaled_dot_kkt_fwd",
    "get_orig_fused_recurrent_gated_delta_rule_packed_decode",
    "get_orig_fused_sigmoid_gating_delta_rule_update",
    "get_orig_solve_tril",
]
