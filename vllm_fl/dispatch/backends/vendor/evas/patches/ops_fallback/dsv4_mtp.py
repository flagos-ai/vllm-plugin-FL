# Copyright (c) 2026 BAAI. All rights reserved.
"""DSV4 MTP draft forward on LPU — rebind the four upstream kernels it imports.

The drafter (``nvidia/mtp.py``) holds its own module-scope copies, so
``model_compat``'s rebinds never reach it: two ``common/ops`` rmsnorms
(row-to-scalar ``tl.sum``, which ``evas-triton-opt`` cannot legalize) get
pure-PyTorch stand-ins through the graduated ``rms_norm`` override on a flat
2-D view, and the two tilelang MHC names reuse what ``model_compat`` already
binds. Idempotent; inert unless the draft runs.

"""

from __future__ import annotations

_GENERIC_PATCHED = False


def _lpu_fused_mtp_input_rmsnorm(
    inputs_embeds,
    positions,
    previous_hidden_states,
    enorm_weight,
    hnorm_weight,
    eps,
    hc_mult,
):
    """LPU stand-in for the upstream ``fused_mtp_input_rmsnorm``.

    Returns ``(enorm_out, hnorm_out)``: ``enorm_out`` shaped like
    ``inputs_embeds`` (2-D ``[T, H]``), ``hnorm_out`` like
    ``previous_hidden_states`` (3-D ``[T, hc_mult, H]``, already reshaped by
    the caller).

    The upstream kernel folds the position-0 mask into the enorm task
    (``keep = pos != 0``) and skips the masked rows' weight, which is exactly
    the eager sequence it replaced: ``where(pos == 0, 0, x)`` followed by a
    norm -- an all-zero row normalizes to zero regardless of weight.  The mask
    is applied before the norm here for the same reason.
    """
    import flag_gems

    hidden = inputs_embeds.shape[-1]
    keep = (positions.reshape(-1, 1) != 0).to(inputs_embeds.dtype)
    masked = inputs_embeds * keep

    enorm_out = flag_gems.rms_norm(masked, (hidden,), enorm_weight, eps)

    prev_2d = previous_hidden_states.reshape(-1, hidden)
    hnorm_out = flag_gems.rms_norm(
        prev_2d, (hidden,), hnorm_weight, eps
    ).reshape(previous_hidden_states.shape)
    return enorm_out, hnorm_out


def _lpu_mtp_shared_head_rmsnorm(hidden_states, weight, eps):
    """LPU stand-in for the upstream ``mtp_shared_head_rmsnorm``.

    Same numeric body as the fused variant's enorm path (upstream shares one
    ``_rmsnorm_row`` between them), on a 2-D ``[T, H]`` input.
    """
    import flag_gems

    hidden = hidden_states.shape[-1]
    out_2d = flag_gems.rms_norm(
        hidden_states.reshape(-1, hidden), (hidden,), weight, eps
    )
    return out_2d.reshape(hidden_states.shape)


def apply_lpu_dsv4_mtp_patch() -> None:
    """Rebind the MTP draft forward's four upstream kernels on LPU.

    Applied unconditionally on LPU: the rebindings are inert unless the MTP
    draft is instantiated, which happens only when speculative decoding enables
    it (the e2e script does that under ``VLLM_FL_DSV4_MTP=1``, an example-script
    knob that no framework code reads).
    """
    global _GENERIC_PATCHED

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return
    if _GENERIC_PATCHED:
        return

    import flag_gems
    import vllm.models.deepseek_v4.nvidia.mtp as _mtp

    from ..model_compat import lpu_hc_head_fused_kernel

    _mtp.fused_mtp_input_rmsnorm = _lpu_fused_mtp_input_rmsnorm
    _mtp.mtp_shared_head_rmsnorm = _lpu_mtp_shared_head_rmsnorm
    _mtp.mhc_post_tilelang = flag_gems.mhc_post
    _mtp.hc_head_fused_kernel_tilelang = lpu_hc_head_fused_kernel

    _GENERIC_PATCHED = True
