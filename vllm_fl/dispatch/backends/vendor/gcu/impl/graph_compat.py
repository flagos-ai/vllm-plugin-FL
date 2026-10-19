# Copyright (c) 2026 BAAI. All rights reserved.

"""Plugin-side monkey-patches to make graph mode (inductor + cudagraph) work
on Enflame GCU300 without modifying site-packages.

Three independent compatibility fixes, each guarded so a single failure only
warns and does not block startup:

A. ``_persistent_reduction_configs`` signature: upstream torch inductor
   triton_heuristics calls it with 4 positional args (incl. ``triton_meta``),
   but torch_gcu's GCUTritonConfigGenerator.persistent_reduction_configs only
   accepts 3 -> TypeError. We wrap it to absorb the extra ``triton_meta``.

B. ``_GCUBackend.parse_options`` enable_i64 fallback: under the inductor path
   neither "ENABLE_I64" nor "enable_i64" is present in ``opts``, so GCUOptions
   defaults enable_i64=False and the i64 verifier rejects dynamic-shape symints
   with "64-bit data type not supported on GCU300". We honor ENABLE_I64_CHECK.

C. flaggems mm autotune configs carry a stray ``SPLIT_K`` kwarg that the
   ``mm_kernel_general`` triton kernel does not accept -> KeyError during
   autotune. We strip SPLIT_K from the (shared) tuned-config objects.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def _patch_persistent_reduction_configs() -> None:
    """A: absorb the extra ``triton_meta`` arg (see module docstring)."""
    try:
        # Force torch_gcu to install its inductor patches first (idempotent via
        # its own _IS_INDUCTOR_PATCHED guard); otherwise it would overwrite our
        # wrapper on the first torch.compile call.
        from torch.gcu.inductor.backend_register import (
            apply_gcu_inductor_monkey_patch,
        )

        apply_gcu_inductor_monkey_patch()

        from torch._inductor.runtime import triton_heuristics

        fn = getattr(triton_heuristics, "_persistent_reduction_configs", None)
        if fn is None:
            logger.warning(
                "graph_compat A: _persistent_reduction_configs not found; "
                "skipping"
            )
            return
        if getattr(fn, "_gcu_graph_wrapped", False):
            return

        @functools.wraps(fn)
        def wrapper(size_hints, reduction_hint=False, inductor_meta=None,
                    triton_meta=None, *args, **kwargs):
            # Drop triton_meta (and any further extras) the GCU impl ignores.
            return fn(size_hints, reduction_hint, inductor_meta)

        wrapper._gcu_graph_wrapped = True
        triton_heuristics._persistent_reduction_configs = wrapper
        logger.info(
            "graph_compat A: wrapped _persistent_reduction_configs to absorb "
            "triton_meta"
        )
    except Exception as exc:
        logger.warning("graph_compat A failed: %s", exc)


def _patch_parse_options_enable_i64() -> None:
    """B: enable_i64 fallback honoring ENABLE_I64_CHECK (see module docstring)."""
    try:
        from triton_gcu.triton import compiler as _gcu_compiler
        from triton_gcu.triton.compiler import _GCUBackend

        if getattr(_GCUBackend.parse_options, "_gcu_graph_patched", False):
            return

        _orig_parse_options = _GCUBackend.parse_options
        toolkit = _gcu_compiler.toolkit

        @functools.wraps(_orig_parse_options)
        def parse_options(self, opts):
            options = _orig_parse_options(self, opts)
            # Only when neither key was present (inductor path) do we fill in
            # the fallback the original two branches miss.
            if "ENABLE_I64" not in opts and "enable_i64" not in opts:
                options.enable_i64 = not toolkit.get_bool_env(
                    "ENABLE_I64_CHECK", True
                )
            return options

        parse_options._gcu_graph_patched = True
        _GCUBackend.parse_options = parse_options
        logger.info(
            "graph_compat B: patched _GCUBackend.parse_options enable_i64 "
            "fallback"
        )
    except Exception as exc:
        logger.warning("graph_compat B failed: %s", exc)


def _patch_flaggems_mm_split_k() -> None:
    """C: strip stray SPLIT_K from flaggems mm tuned configs."""
    try:
        import flag_gems.runtime as rt

        configs = rt.get_tuned_config("mm")
        stripped = 0
        for cfg in configs:
            kw = getattr(cfg, "kwargs", None)
            if isinstance(kw, dict) and "SPLIT_K" in kw:
                kw.pop("SPLIT_K", None)
                stripped += 1
        logger.info(
            "graph_compat C: stripped SPLIT_K from %d/%d flaggems mm configs",
            stripped,
            len(configs),
        )
    except Exception as exc:
        logger.warning("graph_compat C failed: %s", exc)


def apply_graph_mode_compat_patches() -> None:
    """Apply all three graph-mode compatibility patches. Idempotent."""
    _patch_persistent_reduction_configs()
    _patch_parse_options_enable_i64()
    _patch_flaggems_mm_split_k()
