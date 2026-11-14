# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX router bridges for native mcoplib and pure FlagOS images.

vLLM 0.32 passes new trailing arguments to the routed top-k helpers:

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

Images without mcoplib forward the extended router calls to FlagGems-vllm's
public operators. The regular FL softmax route uses CachedOp.

TODO: drop the six-argument bridge once mcoplib ships the extended signatures.
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_metax_moe_topk_compat_patched"
_warned = set()
_topk_operators = None


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

    if importlib.util.find_spec("mcoplib") is None:
        return None
    # Installed-but-broken vendor kernels must surface as a patch error.
    import mcoplib._moe_C  # noqa: F401

    ns = getattr(torch.ops, "_moe_C", None)
    if ns is None or not all(
        hasattr(ns, name) for name in ("topk_softmax", "topk_sigmoid")
    ):
        raise RuntimeError("mcoplib._moe_C does not provide routed top-k operators")
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

    setattr(topk_softmax, _PATCH_FLAG, True)
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

    setattr(topk_sigmoid, _PATCH_FLAG, True)
    return topk_sigmoid


def _topk_softmax_flaggems(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    is_padding=None,
):
    from flaggems_vllm import topk_softmax

    topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias=e_score_correction_bias,
        is_padding=is_padding,
    )


def _topk_sigmoid_flaggems(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    is_padding=None,
):
    from flaggems_vllm import topk_sigmoid

    topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=(
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        ),
        is_padding=is_padding,
    )


setattr(_topk_softmax_flaggems, _PATCH_FLAG, True)
setattr(_topk_sigmoid_flaggems, _PATCH_FLAG, True)


def _make_topk_operators(ns):
    if ns is None:
        return {"softmax": _topk_softmax_flaggems, "sigmoid": _topk_sigmoid_flaggems}
    return {"softmax": _make_topk_softmax(ns), "sigmoid": _make_topk_sigmoid(ns)}


def get_topk_operator(scoring_func):
    """Return our stable callable, independent of _custom_ops rebinding."""
    global _topk_operators
    if _topk_operators is None:
        _topk_operators = _make_topk_operators(_moe_c_namespace())
    return _topk_operators[scoring_func]


def apply():
    """Rebind the routed top-k entry points. Returns True when applied."""
    global _topk_operators
    ns = _moe_c_namespace()
    _topk_operators = _make_topk_operators(ns)
    if ns is not None:
        _force_disable_skip_padding()
    try:
        import vllm._custom_ops as ops
    except Exception as exc:
        logger.warning("metax moe topk compat: vllm._custom_ops unavailable: %s", exc)
        return False
    # vllm._custom_ops calls import_kernels() before defining these functions.
    # Its remaining module body can overwrite our first binding while leaving
    # the module-level flag behind. Only the live function markers establish
    # that both call paths still point to our wrappers.
    if all(
        getattr(getattr(ops, name, None), _PATCH_FLAG, False)
        for name in ("topk_softmax", "topk_sigmoid")
    ):
        return True
    ops.topk_softmax = get_topk_operator("softmax")
    ops.topk_sigmoid = get_topk_operator("sigmoid")
    setattr(ops, _PATCH_FLAG, True)
    logger.info(
        "MetaX moe topk compat applied: vllm._custom_ops.topk_softmax/topk_sigmoid "
        "rebound to %s",
        "FlagGems-vllm public operators"
        if ns is None
        else "6-argument mcoplib _moe_C forms",
    )
    return True


apply()
