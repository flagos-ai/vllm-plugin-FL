# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep Kimi ViT's complex shape operators on the native CUDA implementation.

Capture the PyTorch kernel *before* FlagGems registration. Its documented
SafeKernelFunction API calls that saved implementation without redispatching
back into FlagGems. All real-valued operations retain their original FlagGems callables.
"""

import functools
import logging

logger = logging.getLogger("vllm.fl.kimi_complex_ops")


def _make_cat_dispatch(gems_impl, native_kernel, cuda_keys, *, out_variant=False):
    """Build a metadata-only dtype dispatcher; dependencies permit CPU tests."""
    native_logged = False

    def execute(tensors, dim, out):
        nonlocal native_logged
        complex_input = any(tensor.is_complex() for tensor in tensors)
        complex_output = out is not None and out.is_complex()
        if complex_input or complex_output:
            if out_variant:
                result = native_kernel.call_boxed(cuda_keys, tensors, dim, out=out)
            else:
                result = native_kernel.call_boxed(cuda_keys, tensors, dim)
            if not native_logged:
                logger.info(
                    "FL_KIMI_COMPLEX_CAT_EXECUTED native_cuda=True out=%s tensors=%d",
                    out_variant,
                    len(tensors),
                )
                native_logged = True
            return result
        if out_variant:
            return gems_impl(tensors, dim, out=out)
        return gems_impl(tensors, dim)

    if out_variant:

        @functools.wraps(gems_impl)
        def dispatch(tensors, dim=0, *, out):
            return execute(tensors, dim, out)

    else:

        @functools.wraps(gems_impl)
        def dispatch(tensors, dim=0):
            return execute(tensors, dim, None)

    dispatch._fl_kimi_complex_cat = True
    return dispatch


def _make_slice_dispatch(gems_impl, native_kernel, cuda_keys):
    native_logged = False

    @functools.wraps(gems_impl)
    def dispatch(input_tensor, dim=0, start=None, end=None, step=1):
        nonlocal native_logged
        if not input_tensor.is_complex():
            return gems_impl(input_tensor, dim, start, end, step)
        result = native_kernel.call_boxed(
            cuda_keys, input_tensor, dim, start, end, step
        )
        if not native_logged:
            logger.info("FL_KIMI_COMPLEX_SLICE_EXECUTED native_cuda=True")
            native_logged = True
        return result

    return dispatch


def _make_repeat_dispatch(gems_impl, native_kernel, cuda_keys):
    native_logged = False

    @functools.wraps(gems_impl)
    def dispatch(inp, sizes):
        nonlocal native_logged
        if not inp.is_complex():
            return gems_impl(inp, sizes)
        result = native_kernel.call_boxed(cuda_keys, inp, sizes)
        if not native_logged:
            logger.info("FL_KIMI_COMPLEX_REPEAT_EXECUTED native_cuda=True")
            native_logged = True
        return result

    return dispatch


def patch_kimi_complex_ops(model_config, *, enabled):
    """Call before ``_initialize_fl_runtime`` enables FlagGems on this worker.

    This is scoped to Kimi K3 multimodal workers and complex dtypes. It neither
    changes precision nor skips encoder profiling. Registration uses FlagGems'
    existing live-override resolution; no global Torch API is replaced here.
    """
    if not enabled or getattr(model_config, "architecture", None) != (
        "KimiK3ForConditionalGeneration"
    ):
        return False

    import torch

    # get_kernel is explicitly intended for saving an original implementation
    # before overriding the same dispatch key (PyTorch 2.13 API documentation).
    native = {
        key: torch.library.get_kernel(key, "CUDA")
        for key in ("aten::cat", "aten::cat.out", "aten::slice.Tensor", "aten::repeat")
    }
    cuda_keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)

    import flag_gems

    if getattr(flag_gems.cat, "_fl_kimi_complex_cat", False):
        return False
    registrar = getattr(flag_gems, "current_work_registrar", None)
    if registrar is not None and {"cat", "slice.Tensor", "repeat"}.intersection(
        registrar.get_all_keys()
    ):
        raise RuntimeError("Kimi complex ops must be installed before FlagGems.enable")
    flag_gems.cat = _make_cat_dispatch(flag_gems.cat, native["aten::cat"], cuda_keys)
    flag_gems.cat_out = _make_cat_dispatch(
        flag_gems.cat_out, native["aten::cat.out"], cuda_keys, out_variant=True
    )
    flag_gems.slice = _make_slice_dispatch(
        flag_gems.slice, native["aten::slice.Tensor"], cuda_keys
    )
    flag_gems.repeat = _make_repeat_dispatch(
        flag_gems.repeat, native["aten::repeat"], cuda_keys
    )
    logger.info(
        "FL_KIMI_COMPLEX_OPS_INSTALLED before_flaggems_enable=True "
        "scope=KimiK3ForConditionalGeneration;complex_only"
    )
    return True
