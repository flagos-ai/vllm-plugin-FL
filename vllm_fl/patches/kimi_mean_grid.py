# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep Kimi's oversized single-axis mean off an illegal CUDA grid Y."""

import functools
import logging
import math

logger = logging.getLogger("vllm.fl.kimi_mean_grid")


def _oversized_grid_y(inp, dim, vector_dtypes):
    # Mirrors the pinned Gems mean_dim_comm BF16/FP16 vector branch: grid
    # (prod(shape[:axis]), ceil(prod(shape[axis+1:]) / (128 * 8))).
    # Invalid axes and other dtype/reduction paths retain original validation.
    if inp.dtype not in vector_dtypes:
        return False
    if isinstance(dim, int):
        axes = [dim]
    elif isinstance(dim, (tuple, list)):
        axes = dim
    else:
        return False
    if len(axes) != 1 or not isinstance(axes[0], int):
        return False
    axis = axes[0]
    if not -inp.ndim <= axis < inp.ndim:
        return False
    axis %= inp.ndim
    if inp.shape[axis] == 0:
        return False
    k = math.prod(inp.shape[axis + 1 :])
    return (k + 1023) // 1024 > 65535


def _make_mean_dispatch(gems_impl, native_kernel, cuda_keys, vector_dtypes):
    native_logged = False

    @functools.wraps(gems_impl)
    def dispatch(inp, dim=None, keepdim=False, *, dtype=None):
        nonlocal native_logged
        if not _oversized_grid_y(inp, dim, vector_dtypes):
            return gems_impl(inp, dim, keepdim, dtype=dtype)
        axes = [dim] if isinstance(dim, int) else dim
        result = native_kernel.call_boxed(cuda_keys, inp, axes, keepdim, dtype=dtype)
        if not native_logged:
            logger.info(
                "FL_KIMI_MEAN_GRID_EXECUTED native_cuda=True shape=%s dim=%s "
                "keepdim=%s dtype=%s reason=grid_y_exceeds_65535",
                tuple(inp.shape),
                dim,
                keepdim,
                dtype,
            )
            native_logged = True
        return result

    dispatch._fl_kimi_mean_grid = True
    return dispatch


def patch_kimi_mean_grid(model_config, *, enabled):
    """Capture the native mean kernel before FlagGems enables on Kimi workers."""
    if not enabled or getattr(model_config, "architecture", None) != (
        "KimiK3ForConditionalGeneration"
    ):
        return False

    import flag_gems
    import torch

    if getattr(flag_gems.mean_dim, "_fl_kimi_mean_grid", False):
        return False
    registrar = getattr(flag_gems, "current_work_registrar", None)
    if registrar is not None and "mean.dim" in registrar.get_all_keys():
        raise RuntimeError(
            "Kimi mean grid hook must be installed before FlagGems.enable"
        )
    native = torch.library.get_kernel("aten::mean.dim", "CUDA")
    cuda_keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    flag_gems.mean_dim = _make_mean_dispatch(
        flag_gems.mean_dim, native, cuda_keys, (torch.float16, torch.bfloat16)
    )
    logger.info(
        "FL_KIMI_MEAN_GRID_INSTALLED before_flaggems_enable=True "
        "scope=KimiK3ForConditionalGeneration;half_single_axis_grid_y_gt65535"
    )
    return True
