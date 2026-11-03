# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX C550 compatibility shim for the vLLM fused-MoE top-k routers.

vLLM 0.28 appends new trailing arguments to the routed top-k helpers:

  vllm._custom_ops.topk_softmax(..., renormalize, e_score_correction_bias, is_padding)
  vllm._custom_ops.topk_sigmoid(..., renormalize, e_score_correction_bias,
                                routed_scaling_factor, is_padding)

mcoplib for MACA only implements the 6-argument forms:

  _moe_C::topk_softmax(Tensor, Tensor, Tensor, Tensor, bool, Tensor?) -> ()
  _moe_C::topk_sigmoid(Tensor, Tensor, Tensor, Tensor, bool, Tensor?) -> ()

so every MoE model dies during profile_run with

  _moe_C::topk_softmax() expected at most 6 argument(s) but received 7 argument(s).

This module rebinds the two entry points on the vllm._custom_ops module object.
Callers reach them through that module (import vllm._custom_ops as ops in
vllm/model_executor/layers/fused_moe/router/*, and
from vllm._custom_ops import topk_softmax in the MetaX vendor backend), so a
single module-attribute rebind covers every call path.

TODO: drop once mcoplib ships the extended signatures.
"""

import logging
import os

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_metax_moe_topk_compat_patched"
_warned = set()


def _warn_once(key, msg, *args):
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def _force_disable_skip_padding():
    raw = os.environ.get("VLLM_MOE_SKIP_PADDING", "")
    if str(raw).strip().lower() not in ("", "0", "false", "no", "off"):
        _warn_once(
            "skip_padding",
            "VLLM_MOE_SKIP_PADDING=%s is unsupported on MetaX (mcoplib topk kernels "
            "take no is_padding argument); forcing it off.",
            raw,
        )
    os.environ["VLLM_MOE_SKIP_PADDING"] = "0"


def _moe_c_namespace():
    import torch

    try:
        import mcoplib._moe_C  # noqa: F401
    except Exception as exc:
        logger.warning("metax moe topk compat: mcoplib._moe_C import failed: %s", exc)

    ns = getattr(torch.ops, "_moe_C", None)
    if ns is None or not hasattr(ns, "topk_softmax"):
        return None
    return ns


def _make_topk_softmax(ns):
    def topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        is_padding=None,
    ):
        if is_padding is not None:
            _warn_once(
                "softmax_padding",
                "metax topk compat: dropping is_padding for _moe_C.topk_softmax",
            )
        ns.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
        )

    return topk_softmax


def _make_topk_sigmoid(ns):
    def topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        is_padding=None,
    ):
        if is_padding is not None:
            _warn_once(
                "sigmoid_padding",
                "metax topk compat: dropping is_padding for _moe_C.topk_sigmoid",
            )
        ns.topk_sigmoid(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
        )
        if routed_scaling_factor is not None:
            factor = float(routed_scaling_factor)
            if factor != 1.0:
                topk_weights.mul_(factor)

    return topk_sigmoid


def apply():
    """Rebind the routed top-k entry points. Returns True when applied."""
    _force_disable_skip_padding()
    ns = _moe_c_namespace()
    if ns is None:
        logger.warning(
            "metax moe topk compat: mcoplib _moe_C topk ops not found; leaving "
            "vllm._custom_ops untouched"
        )
        return False
    try:
        import vllm._custom_ops as ops
    except Exception as exc:
        logger.warning("metax moe topk compat: vllm._custom_ops unavailable: %s", exc)
        return False
    if getattr(ops, _PATCH_FLAG, False):
        return True
    ops.topk_softmax = _make_topk_softmax(ns)
    ops.topk_sigmoid = _make_topk_sigmoid(ns)
    setattr(ops, _PATCH_FLAG, True)
    logger.info(
        "MetaX moe topk compat applied: vllm._custom_ops.topk_softmax/topk_sigmoid "
        "rebound to the 6-argument mcoplib _moe_C forms"
    )
    return True


apply()
