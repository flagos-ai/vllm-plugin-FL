# Copyright (c) 2026 BAAI. All rights reserved.
"""Install large-M mHC/norm fusion only for the validated RMSNorm backend."""

import os
from types import MethodType

import torch


def _pre(layer, x, branch):
    norm = getattr(layer, branch + "_norm")
    return torch.ops.vllm.fl_hygon_dv4_prefill_mhc_pre_norm(
        x, getattr(layer, "hc_" + branch + "_fn"),
        getattr(layer, "hc_" + branch + "_scale"),
        getattr(layer, "hc_" + branch + "_base"), norm.weight,
        layer.rms_norm_eps, layer.hc_eps, layer.hc_eps,
        layer.hc_post_alpha, layer.hc_sinkhorn_iters, norm.variance_epsilon,
    )


def _forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
    # M is the scheduled token count, including mixed decode tokens. Small
    # batches retain exactly the existing decoder implementation.
    if x.shape[0] < 1024 or x.dtype != torch.bfloat16:
        return self._hygon_original_mhc_forward(
            x, positions, input_ids, post_mix, res_mix, residual)
    if self.has_tilelang and residual is not None:
        residual = self.mhc_post(x, residual, post_mix, res_mix)
    else:
        residual = x
    post_mix, res_mix, x = _pre(self, residual, "attn")
    x = self.attn(positions, x, None)
    residual = self.mhc_post(x, residual, post_mix, res_mix)
    post_mix, res_mix, x = _pre(self, residual, "ffn")
    x = self.ffn(x, input_ids)
    if self.has_tilelang:
        return x, residual, post_mix, res_mix
    return self.mhc_post(x, residual, post_mix, res_mix), None, None, None


def install_prefill_mhc_norm(layer):
    if os.getenv("VLLM_FL_HYGON_PREFILL_MHC_NORM", "1").lower() in ("0", "off", "false"):
        return False
    if hasattr(layer, "_hygon_original_mhc_forward"):
        return True
    from flag_gems.config import use_c_extension
    from vllm_fl.dispatch import get_default_manager
    from vllm_fl.ops.layernorm import RMSNormFL

    # Do not bypass a user-selected LightOp/reference RMSNorm, variance-size
    # override, or the separate C-extension implementation.
    candidates = get_default_manager().resolve_candidates("rms_norm")
    if use_c_extension or not candidates or candidates[0].impl_id != "default.flagos":
        return False
    if getattr(layer, "hidden_size", None) != 4096:
        return False
    for norm in (layer.attn_norm, layer.ffn_norm):
        if (type(norm) is not RMSNormFL or norm.variance_size_override is not None
                or norm.weight.shape != (4096,) or not norm.has_weight
                or getattr(norm._forward_method, "__func__", None) is not RMSNormFL.forward_oot):
            return False

    # Register the opaque op before Dynamo sees the decoder forward.
    from vllm_fl.dispatch.backends.vendor.hygon.impl.other import prefill_mhc  # noqa: F401

    object.__setattr__(layer, "_hygon_original_mhc_forward", layer.forward)
    object.__setattr__(layer, "forward", MethodType(_forward, layer))
    return True
