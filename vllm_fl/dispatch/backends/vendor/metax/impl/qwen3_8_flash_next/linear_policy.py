# SPDX-License-Identifier: Apache-2.0
"""Narrow native primitive for reviewed GR down projection; caller owns policy.
Preserves FP32 accumulation followed by BF16 storage before hc_count division.
No use_gems context switch, no sampling change, no library registration edits.
Small-M eager is slower; the parent policy must explicitly select this for its
measured graph-dominated workload. Use the same path in warmup and capture.
"""

import torch


def selected_rows(rows):
    return 1 <= rows <= 64 or 1024 <= rows <= 2048


def supports_native_grdown(x, weight):
    return (
        x.ndim == 2
        and weight.ndim == 2
        and selected_rows(x.shape[0])
        and x.shape[1] == 10240
        and tuple(weight.shape) == (320, 10240)
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.device.type == "cuda"
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def native_grdown(x, weight):
    if not supports_native_grdown(x, weight):
        raise ValueError(
            "Unsupported GR down shape; caller must retain its normal path"
        )
    return torch.ops.aten.mm.dtype(x, weight.t(), out_dtype=torch.float32).to(
        torch.bfloat16
    )
