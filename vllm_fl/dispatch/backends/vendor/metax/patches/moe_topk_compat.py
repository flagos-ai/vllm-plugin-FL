# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX router bridges for native mcoplib and pure FlagOS images.

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

Images without mcoplib use Torch for the native router's padding, sigmoid and
correction-bias paths. The regular FL softmax route still uses CachedOp.

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


def _topk_torch(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize,
    e_score_correction_bias,
    is_padding,
    *,
    sigmoid=False,
    routed_scaling_factor=1.0,
):
    """Bridge the native router ABI when a FlagOS image has no mcoplib.

    The ordinary FL softmax route still uses CachedOp. This path covers
    padding, correction bias and sigmoid routes delegated to vLLM's helpers.
    Selection and expanded row indices follow csrc/moe/topk_softmax_kernels.cu.
    """
    import torch

    logits = gating_output.float()
    scores = torch.sigmoid(logits) if sigmoid else torch.softmax(logits, dim=-1)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    choices = scores
    if e_score_correction_bias is not None:
        choices = choices + e_score_correction_bias.float()
    # The native argmax breaks ties toward the lower expert ID. torch.topk
    # does not guarantee that ordering; stable argsort does.
    selected = torch.argsort(choices, dim=-1, descending=True, stable=True)[
        :, : topk_weights.shape[-1]
    ]
    weights = scores.gather(1, selected)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    if routed_scaling_factor is not None:
        weights = weights * routed_scaling_factor
    if is_padding is not None:
        padding = is_padding.reshape(-1, 1)
        weights = torch.where(padding, 0.0, weights)
        selected = torch.where(padding, -1, selected)
    topk_weights.copy_(weights)
    topk_ids.copy_(selected)
    num_tokens, top_k = topk_weights.shape
    rows = torch.arange(
        num_tokens, dtype=token_expert_indices.dtype, device=gating_output.device
    )[:, None]
    slots = torch.arange(
        top_k, dtype=token_expert_indices.dtype, device=gating_output.device
    )[None, :]
    token_expert_indices.copy_(slots * num_tokens + rows)


def _topk_softmax_torch(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    is_padding=None,
):
    _topk_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding,
    )


def _topk_sigmoid_torch(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    is_padding=None,
):
    _topk_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding,
        sigmoid=True,
        routed_scaling_factor=routed_scaling_factor,
    )


setattr(_topk_softmax_torch, _PATCH_FLAG, True)
setattr(_topk_sigmoid_torch, _PATCH_FLAG, True)


def get_topk_operator(scoring_func):
    """Return our stable callable, independent of _custom_ops rebinding."""
    global _topk_operators
    if _topk_operators is None:
        ns = _moe_c_namespace()
        _topk_operators = {
            "softmax": _topk_softmax_torch if ns is None else _make_topk_softmax(ns),
            "sigmoid": _topk_sigmoid_torch if ns is None else _make_topk_sigmoid(ns),
        }
    return _topk_operators[scoring_func]


def apply():
    """Rebind the routed top-k entry points. Returns True when applied."""
    global _topk_operators
    ns = _moe_c_namespace()
    _topk_operators = {
        "softmax": _topk_softmax_torch if ns is None else _make_topk_softmax(ns),
        "sigmoid": _topk_sigmoid_torch if ns is None else _make_topk_sigmoid(ns),
    }
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
        "Torch FlagOS fallback" if ns is None else "6-argument mcoplib _moe_C forms",
    )
    return True


apply()
