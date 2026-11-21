# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX INT8 Linear bridge to the existing public FGV scaled GEMM."""


def is_available():
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return False
    import flaggems_vllm

    return callable(getattr(flaggems_vllm, "triton_scaled_mm", None))


def w8a8_scaled_mm(a, b, scale_a, scale_b, *, bias=None, out_dtype):
    import torch
    from flaggems_vllm import triton_scaled_mm

    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("MetaX W8A8 scaled GEMM requires INT8 operands")
    return triton_scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
        bias=bias,
        block_size_m=16,
        block_size_n=32,
        block_size_k=128,
        use_heuristic=False,
    )
