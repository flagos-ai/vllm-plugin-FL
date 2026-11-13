# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX filter selection with unchanged native full-vocabulary RNG."""

from contextvars import ContextVar
from functools import wraps

import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

from vllm_fl import envs as fl_envs

_POSITIVE_NATIVE_NOISE = ContextVar("metax_positive_native_noise", default=False)


def _original(function):
    return (
        function.__wrapped__
        if getattr(function, "_fl_metax_sampling_patch", False)
        else function
    )


_native_random_sample = _original(topk_topp_sampler.random_sample)
_native_sample_with_noise = _original(topk_topp_sampler.sample_with_exponential_noise)


def _supported(x):
    return (
        x.device.type == "cuda"
        and x.dtype == torch.float32
        and x.ndim == 2
        and 1 <= x.shape[0] <= 64
        and x.shape[1] == 248320
        and x.is_contiguous()
    )


def _apply_top_k_top_p_no_triton(logits, k, p):
    if p is None and k is None:
        return logits
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        if (
            _supported(logits)
            and k is not None
            and getattr(k, "_fl_all_topk20", False)
            and p is not None
            and p.dtype == torch.float32
            and p.device == logits.device
            and p.shape == (logits.shape[0],)
            and p.is_contiguous()
        ):
            from flaggems_vllm.runtime.backend._metax.ops.sampling_top20 import (
                apply_top20_top_p,
            )

            return apply_top20_top_p(logits, p)
        import flaggems_vllm

        apply_filter = getattr(flaggems_vllm, "apply_top_k_top_p", None)
        if apply_filter is not None:
            return apply_filter(logits, k, p)
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


@wraps(_native_random_sample)
def _random_sample(probs, generators, use_fp64_gumbel=False):
    enabled = (
        fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM
        and _supported(probs)
        and not generators
        and not use_fp64_gumbel
    )
    token = _POSITIVE_NATIVE_NOISE.set(enabled)
    try:
        # Original allocation, full-vocabulary exponential draws and RNG state.
        return _native_random_sample(probs, generators, use_fp64_gumbel)
    finally:
        _POSITIVE_NATIVE_NOISE.reset(token)


@wraps(_native_sample_with_noise)
def _sample_with_exponential_noise(probs, noise):
    if (
        _POSITIVE_NATIVE_NOISE.get()
        and _supported(probs)
        and noise.dtype == probs.dtype
        and noise.shape == probs.shape
        and noise.device == probs.device
        and noise.is_contiguous()
    ):
        from flaggems_vllm.runtime.backend._metax.ops.sampling_race import sample

        return sample(probs, noise, tile=2048, warps=4)
    return _native_sample_with_noise(probs, noise)


def install_metax_sampling_patches():
    """Publish the filter and paired RNG hooks once, without partial setup."""
    from vllm_fl.patches._qwen38_installation import apply_attributes

    changes = (
        ("apply_top_k_top_p", _apply_top_k_top_p_no_triton),
        ("random_sample", _random_sample),
        ("sample_with_exponential_noise", _sample_with_exponential_noise),
    )
    for name, _ in changes:
        if not callable(getattr(topk_topp_sampler, name, None)):
            raise TypeError(f"MetaX sampling requires native {name}")
    if all(
        getattr(getattr(topk_topp_sampler, name), "_fl_metax_sampling_patch", False)
        for name, _ in changes
    ):
        return
    for _, function in changes:
        function._fl_metax_sampling_patch = True
    apply_attributes(
        [(topk_topp_sampler, name, function) for name, function in changes]
    )
