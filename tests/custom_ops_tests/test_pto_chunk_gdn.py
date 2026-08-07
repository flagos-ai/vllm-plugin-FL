# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal integration test for the PTO GDN megakernel.
# Verifies that the 6-stage fused AscendC/PTO kernel can be compiled and
# launched through vllm-plugin-FL's Python bindings.

import os
import sys
import tempfile

import torch
import torch_npu

# Allow importing the in-tree vllm_fl without installation.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Load mega_kernel via importlib, bypassing
# vllm_fl.dispatch.backends.vendor.ascend.impl.__init__ (which eagerly
# imports modules that require a fully-initialized vLLM runtime).
#
# We register the module under a dotted package name so that the relative
# import "from .compile import ..." inside mega_kernel.py resolves correctly
# against sibling modules registered in sys.modules under the same package.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_PTO_PKG = "vllm_fl.dispatch.backends.vendor.ascend.impl.pto_chunk_gdn"
_PTO_DIR = os.path.join(
    ROOT,
    "vllm_fl", "dispatch", "backends", "vendor", "ascend", "impl", "pto_chunk_gdn",
)


def _load_sibling(name, path):
    """Load a Python file as a module of package _PTO_PKG."""
    spec = _ilu.spec_from_file_location(f"{_PTO_PKG}.{name}", path)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = _PTO_PKG
    sys.modules[f"{_PTO_PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load compile.py first (dependency of mega_kernel.py), then mega_kernel.py.
_load_sibling("compile", os.path.join(_PTO_DIR, "compile.py"))
_mega = _load_sibling("mega_kernel", os.path.join(_PTO_DIR, "mega_kernel.py"))
run_mega_kernel = _mega.run_mega_kernel


def main() -> int:
    torch.npu.config.allow_internal_format = True

    from vllm_fl.utils import enable_custom_op

    enable_custom_op()

    device = torch.device("npu:0")
    B, T, Hg, H, D = 1, 128, 8, 16, 128

    q = torch.randn(B, T, Hg, D, dtype=torch.float16, device=device)
    k = torch.randn(B, T, Hg, D, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float16, device=device)
    g_in = torch.randn(B, T, H, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    stream = torch.npu.current_stream()._as_parameter_

    print("Compiling / loading PTO GDN megakernel ...")
    try:
        out = run_mega_kernel(
            q,
            k,
            v,
            g_in,
            beta,
            cu_seqlens,
            stream=stream,
            chunk_size=128,
            scale=D**-0.5,
            key_heads=Hg,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    expected_shape = (B, T, H, D)
    if out.shape != expected_shape:
        print(f"FAIL: output shape {tuple(out.shape)} != expected {expected_shape}")
        return 1

    print(f"OK: PTO GDN megakernel produced output shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())