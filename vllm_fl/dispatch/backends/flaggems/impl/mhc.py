# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems implementations for mHC (Multi-Head Convolution) operators.

All imports and control logic are hoisted to module scope: the frozen
CachedOp dispatch calls these functions from inside ``torch.compile`` traces,
so no lazy import / monkey-patching may happen in the call path (the first
trace would fail at ``importlib`` machinery inside Dynamo).

Execution policy:

- PT2-capable builds (``torch.library.triton_op`` + ``wrap_triton`` present)
  route through ``flag_gems.pt2.mhc``, which launches the *original* FlagGems
  JITFunctions traceably in both eager and compiled execution.  The GEMM then
  performs a plain ``fn.to(bfloat16)`` — the same aten op the eager
  ``_FN_BF16_CACHE`` would return — so the WeakKeyDictionary/id cache and its
  patch never enter the traced path at all, making that patch unnecessary on
  this path.
- Older Torch builds (USE_C_EXTENSION=1-style eager environments) keep the
  exact pre-existing eager behavior: ``flag_gems.mhc_*`` with the
  ``_FN_BF16_CACHE`` id-key patch applied at import time.
"""

import torch

import flag_gems
import flag_gems.fused.mhc as _mhc_pkg  # noqa: F401  (ensures submodules loaded)
from flag_gems.pt2 import mhc as _pt2_mhc

# ``import flag_gems.fused.mhc.mhc_pre as ...`` would bind the re-exported
# *function* (flag_gems.fused.mhc.__init__ shadows the submodule), so go
# through importlib to get the module object itself.
import importlib as _importlib

_mhc_pre_mod = _importlib.import_module("flag_gems.fused.mhc.mhc_pre")

_USE_PT2_MHC = _pt2_mhc.supports_pt2_triton()

if not _USE_PT2_MHC:
    # Workaround: flag_gems uses a WeakKeyDictionary with tensor keys.
    # WeakKeyDictionary.get() creates a new weakref and dict lookup calls
    # ref.__eq__ which delegates to tensor.__eq__, returning a multi-element
    # tensor instead of a scalar bool — causing RuntimeError.
    # Patch the module's _FN_BF16_CACHE with an id-based cache.  Applied once
    # at import time, and only on the eager path that actually calls
    # ``flag_gems.fused.mhc.mhc_pre`` directly.

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

    _patch_fn_bf16_cache(_mhc_pre_mod)


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
    if _USE_PT2_MHC:
        return _pt2_mhc.mhc_pre(
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

    return flag_gems.mhc_pre(
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


def mhc_post_flaggems(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """FlagGems native implementation of mhc_post."""
    if _USE_PT2_MHC:
        return _pt2_mhc.mhc_post(x, residual, post, comb)

    return flag_gems.mhc_post(x, residual, post, comb)


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
    if _USE_PT2_MHC:
        _pt2_mhc.hc_head_fused_kernel(
            hs_flat, fn, hc_scale, hc_base, out,
            hidden_size, rms_eps, hc_eps, hc_mult,
        )
        return

    flag_gems.hc_head_fused_kernel(
        hs_flat, fn, hc_scale, hc_base, out,
        hidden_size, rms_eps, hc_eps, hc_mult,
    )
