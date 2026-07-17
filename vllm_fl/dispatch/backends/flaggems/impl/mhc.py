# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems implementations for mHC (Multi-Head Convolution) operators.
"""

import torch


def mhc_pre_flaggems(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlagGems native implementation of mhc_pre."""
    import importlib
    from flag_gems import mhc_pre

    # Workaround: flag_gems uses a WeakKeyDictionary with tensor keys.
    # WeakKeyDictionary.get() creates a new weakref and dict lookup calls
    # ref.__eq__ which delegates to tensor.__eq__, returning a multi-element
    # tensor instead of a scalar bool — causing RuntimeError.
    # Patch the module's _FN_BF16_CACHE with an id-based cache.
    _mhc_pre_mod = importlib.import_module('flag_gems.fused.mhc.mhc_pre')
    _patch_fn_bf16_cache(_mhc_pre_mod)

    return mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=n_splits,
    )


def _patch_fn_bf16_cache(mod):
    """Replace the WeakKeyDictionary-based _FN_BF16_CACHE with an id-keyed dict."""
    if getattr(mod, '_FN_BF16_CACHE_PATCHED', False):
        return

    class _IdKeyCache:
        """Cache keyed by tensor id (data_ptr) to avoid tensor __eq__ issues."""

        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(id(key), default)

        def __setitem__(self, key, value):
            self._data[id(key)] = value

        def __getitem__(self, key):
            return self._data[id(key)]

        def __contains__(self, key):
            return id(key) in self._data

    mod._FN_BF16_CACHE = _IdKeyCache()
    mod._FN_BF16_CACHE_PATCHED = True


def mhc_post_flaggems(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """FlagGems native implementation of mhc_post."""
    from flag_gems import mhc_post

    return mhc_post(x, residual, post, comb)


def hc_head_fused_kernel_flaggems(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """FlagGems native implementation of hc_head_fused_kernel. Mutates `out` in-place."""
    from flag_gems import hc_head_fused_kernel

    hc_head_fused_kernel(
        hs_flat, fn, hc_scale, hc_base, out,
        hidden_size, rms_eps, hc_eps, hc_mult,
    )
