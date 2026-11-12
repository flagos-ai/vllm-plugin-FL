# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Model-scoped Linear launch selection; numerical calls are public FG APIs."""

from collections.abc import Mapping
from copy import deepcopy

import torch
from flag_gems.ops.linear import linear as native_linear

from vllm_fl.dispatch.backends.flaggems.impl.minimax_m3.linear import linear_with_config

_POLICY = None
_SMALL = {}
_LEGACY = {}
_ORDERED = {}


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(v) for v in value]
    return value


def configure(policy):
    """Install explicit shape rules before serving; no model/parallel imports."""
    global _POLICY, _SMALL, _LEGACY, _ORDERED
    policy = deepcopy(thaw(policy))
    if _POLICY is not None and policy != _POLICY:
        raise RuntimeError("Linear launch policy is already configured")
    _POLICY = policy
    _SMALL = {
        (r["M"], r["N"], r["K"], r["dtype"]): r["candidate"] for r in policy["small"]
    }
    _LEGACY = {
        (r["M"], r["N"], r["K"], r["dtype"]): r["config"] for r in policy["legacy"]
    }
    _ORDERED = {
        (r["M"], r["N"], r["K"]): {k: r[k] for k in ("ORDER", "GROUP_M")}
        for r in policy["ordered_shapes"]
    }


def is_configured():
    return _POLICY is not None


def linear(input, weight, bias=None):
    flat = input.unsqueeze(0) if input.dim() == 1 else input
    m = 1
    for dim in flat.shape[:-1]:
        m *= dim
    k, n = flat.shape[-1], weight.shape[0]
    a = flat.view(m, k)
    w = weight.contiguous()
    if _POLICY is None:
        return native_linear(input, w, bias)
    key = (m, n, k, str(a.dtype).split(".")[-1])
    small = _SMALL.get(key)
    same = a.is_cuda and a.device == w.device and a.dtype == w.dtype
    contiguous = a.is_contiguous() and w.is_contiguous()
    if (
        small is not None
        and bias is None
        and same
        and contiguous
        and not (a.requires_grad or w.requires_grad)
    ):
        return linear_with_config(
            input,
            w,
            config=thaw(small["config"]),
            kind=small["kind"],
            order=small.get("order", 0),
            group_m=small.get("group_m", 4),
        )
    order = _ORDERED.get((m, n, k))
    if (
        order is not None
        and bias is None
        and same
        and contiguous
        and a.dtype == torch.bfloat16
    ):
        return linear_with_config(
            input,
            w,
            config=thaw(_POLICY["ordered_config"]),
            order=order["ORDER"],
            group_m=order["GROUP_M"],
        )
    config = _LEGACY.get(key)
    if config is not None and bias is None and same and contiguous:
        return linear_with_config(input, w, config=thaw(config))
    return native_linear(input, w, bias)
