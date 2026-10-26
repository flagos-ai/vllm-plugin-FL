# Copyright (c) 2026 BAAI. All rights reserved.
"""Compatibility metadata for vLLM 0.28's Triton JIT monitor."""


def patch_triton_compile_hooks() -> None:
    from triton.backends.tsingmicro.compiler import TXDAOptions

    # The vendor Triton hook formatter reads this CUDA option even for TXDA.
    # Keep it as a class attribute: dataclass fields, instance __dict__,
    # compiler options and cache hashes must remain unchanged.
    if not hasattr(TXDAOptions, "launch_cooperative_grid"):
        TXDAOptions.launch_cooperative_grid = False


patch_triton_compile_hooks()
