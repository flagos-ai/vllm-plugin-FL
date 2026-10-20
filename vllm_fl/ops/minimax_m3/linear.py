# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Shape-scoped launch policy; no runtime function replacement or framework imports."""

from collections.abc import Mapping
from copy import deepcopy

import torch
import triton
from flag_gems.ops.linear import linear_kernel as _native_kernel
from flag_gems.ops.linear_primitives import (
    linear_gemv_untransposed_kernel as linear_gemv,
)

from .jit import launch_handle

linear_scheduled = launch_handle(_native_kernel)
legacy_kernel = linear_scheduled

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


_COUNTS = {"small": 0, "ordered": 0, "legacy": 0, "fallback": 0}


def snapshot():
    return dict(_COUNTS)


def launch(native, grid, a, w, bias, out, M, N, K, *strides, **kwargs):
    if _POLICY is None:
        return native[grid](a, w, bias, out, M, N, K, *strides, **kwargs)
    dtype = str(a.dtype).split(".")[-1]
    key = (M, N, K, dtype)
    small = _SMALL.get(key)
    small_ok = (
        all(isinstance(v, int) for v in (M, N, K))
        and not (set(kwargs) - {"BIAS"})
        and not kwargs.get("BIAS", False)
        and all(isinstance(x, torch.Tensor) for x in (a, w, out))
        and a.is_cuda
        and a.device == w.device == out.device
        and a.dtype == w.dtype == out.dtype
        and not any(x.requires_grad for x in (a, w, out))
        and a.shape == (M, K)
        and w.shape == (N, K)
        and out.shape == (M, N)
        and all(x.is_contiguous() for x in (a, w, out))
        and tuple(strides) == (K, 1, K, 1, N, 1, 0)
    )
    if small is not None and small_ok:
        _COUNTS["small"] += 1
        cfg = thaw(small["config"])
        if small["kind"] == "gemv":
            return linear_gemv[(triton.cdiv(M, cfg["BM"]), triton.cdiv(N, cfg["BN"]))](
                a, w, out, M, N, K, **cfg
            )
        order = small.get("order", 0)
        gm, gn = (
            triton.cdiv(M, cfg["BLOCK_SIZE_M"]),
            triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
        )
        return linear_scheduled[(gm * gn,) if order else (gm, gn)](
            a,
            w,
            bias,
            out,
            M,
            N,
            K,
            *strides,
            BIAS=False,
            ORDER=order,
            GROUP_M=small.get("group_m", 4),
            **cfg,
        )
    order = _ORDERED.get((M, N, K))
    if (
        order is not None
        and not kwargs.get("BIAS", False)
        and a.dtype == w.dtype == out.dtype == torch.bfloat16
        and a.is_cuda
        and a.device == w.device == out.device
        and all(x.is_contiguous() for x in (a, w, out))
        and tuple(strides[:6]) == (K, 1, K, 1, N, 1)
        and not (set(kwargs) - {"BIAS"})
    ):
        _COUNTS["ordered"] += 1
        cfg = thaw(_POLICY["ordered_config"])
        count = (M // cfg["BLOCK_SIZE_M"]) * (N // cfg["BLOCK_SIZE_N"])
        return linear_scheduled[(count,)](
            a,
            w,
            bias,
            out,
            M,
            N,
            K,
            *strides,
            **kwargs,
            **cfg,
            **order,
        )
    config = _LEGACY.get(key)
    if (
        config is not None
        and not kwargs.get("BIAS", False)
        and a.is_contiguous()
        and w.is_contiguous()
        and out.is_contiguous()
        and w.dtype == a.dtype
        and out.dtype == a.dtype
    ):
        _COUNTS["legacy"] += 1
        cfg = thaw(config)
        return legacy_kernel[
            (triton.cdiv(M, cfg["BLOCK_SIZE_M"]), triton.cdiv(N, cfg["BLOCK_SIZE_N"]))
        ](a, w, bias, out, M, N, K, *strides, **kwargs, **cfg)
    _COUNTS["fallback"] += 1
    return native[grid](a, w, bias, out, M, N, K, *strides, **kwargs)


def linear(input, weight, bias=None):
    """ATen registration scoped to MiniMax; preserve the pinned Linear wrapper."""
    batch_dims = input.shape[:-1]
    one_dim = input.dim() == 1
    flat = input.unsqueeze(0) if one_dim else input
    dims = flat.shape[:-1]
    m = 1
    for dim in dims:
        m *= dim
    k, n = flat.shape[-1], weight.shape[0]
    a = flat.view(m, k)
    w = weight.contiguous()
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]),
        triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device(input.device):
        launch(
            _native_kernel,
            grid,
            a,
            w,
            bias if bias is not None else w,
            out,
            m,
            n,
            k,
            *a.stride(),
            *w.stride(),
            *out.stride(),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
        )
    return out.view(*batch_dims, n)
