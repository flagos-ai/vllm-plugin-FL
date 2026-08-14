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


def main() -> int:
    torch.npu.config.allow_internal_format = True

    from vllm_fl.utils import enable_custom_op

    enable_custom_op()

    from vllm_fl.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel

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
