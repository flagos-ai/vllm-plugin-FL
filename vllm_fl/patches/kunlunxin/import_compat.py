# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""Import-time compatibility for the Kunlunxin runtime.

Keep these process-wide mutations in the Kunlunxin-specific tree.  The shared
package initializer is responsible only for deciding whether this module may
run, before importing FlagGems or probing accelerator runtimes.

This module intentionally lives below the otherwise lightweight
``vllm_fl.patches`` package.  Importing the normal dispatch vendor path would
load dispatch policy, which imports FlagGems before this compatibility shim
has had a chance to run.
"""

import importlib
import sys
import types


def _patch_flag_gems_triton_import_compat() -> None:
    """Allow newer FlagGems to load with the Kunlunxin Triton runtime.

    FlagGems 5.4 registers ``_dirichlet_grad`` at import time and asks Triton
    to resolve ``tl.map_elementwise`` while computing the JIT cache key.  The
    Kunlunxin Triton runtime does not expose that builtin.  vLLM does not use
    this operator and the Kunlunxin dispatch config blacklists it, so provide
    only an import-time sentinel.  If it is ever invoked, fail explicitly
    instead of silently producing an incorrect result.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return

    if not hasattr(tl, "map_elementwise"):

        def _unsupported_map_elementwise(*args, **kwargs):
            raise NotImplementedError(
                "triton.language.map_elementwise is unavailable on Kunlunxin; "
                "the FlagGems _dirichlet_grad operator must remain blacklisted"
            )

        _unsupported_map_elementwise.__name__ = "map_elementwise"
        _unsupported_map_elementwise.__module__ = "triton.language"
        _unsupported_map_elementwise.__triton_builtin__ = True
        tl.map_elementwise = _unsupported_map_elementwise

    try:
        importlib.import_module("triton.knobs")
    except ModuleNotFoundError as exc:
        if exc.name != "triton.knobs":
            raise

        knobs = types.ModuleType("triton.knobs")
        knobs.autotuning = types.SimpleNamespace(adjust_block_size=True)
        sys.modules[knobs.__name__] = knobs
        triton.knobs = knobs


def _patch_torch_float4_import_compat() -> None:
    """Provide the dtype sentinel expected by vLLM on Kunlunxin Torch builds."""
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
    else:
        import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.uint8


def apply_import_compat() -> None:
    """Apply all Kunlunxin shims required before importing FlagGems or vLLM."""
    _patch_flag_gems_triton_import_compat()
    _patch_torch_float4_import_compat()
