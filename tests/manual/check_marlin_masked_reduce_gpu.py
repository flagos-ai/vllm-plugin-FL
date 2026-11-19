# SPDX-License-Identifier: Apache-2.0
"""Isolated masked reduction numerical and changed-mask graph checks."""

import importlib
import json
from unittest.mock import patch

import torch

from vllm_fl.ops.fused_moe.mxfp4_marlin_reduce import masked_moe_sum, safe_moe_sum

kernel_module = importlib.import_module(
    "vllm_fl.ops.fused_moe.mxfp4_marlin_masked_reduce"
)
with patch.object(
    kernel_module, "masked_sum", wraps=kernel_module.masked_sum
) as called:
    x = torch.randn((16, 6, 128), device="cuda", dtype=torch.bfloat16)
    mask = torch.ones((16, 6), device="cuda", dtype=torch.bool)
    y = torch.empty((16, 128), device="cuda", dtype=x.dtype)
    masked_moe_sum(importlib.import_module("flaggems_vllm"), x, mask, y)
    assert called.call_count == 1
gems = importlib.import_module("flaggems_vllm")
torch.manual_seed(44834943)
records = []
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for m, k, h in [
        (0, 6, 128),
        (1, 6, 4096),
        (64, 6, 4096),
        (128, 6, 4096),
        (4096, 6, 4096),
        (65536, 6, 128),
        (16, 1, 256),
        (16, 8, 256),
        (16, 7, 256),
        (1, 3, 128),
        (129, 7, 256),
        (128, 6, 130),
    ]:
        x = torch.randn((m, k, h), device="cuda", dtype=dtype)
        mask = torch.rand((m, k), device="cuda") > 0.4
        original = x.clone()
        actual = torch.empty((m, h), device="cuda", dtype=dtype)
        ref = torch.empty_like(actual)
        for phase in range(3):
            mask.copy_(
                (torch.arange(m * k, device="cuda").reshape(m, k) + phase) % 3 != 0
            )
            x.copy_(original + 0.1 * phase)
            x.masked_fill_(
                ~mask[:, :, None],
                float("nan")
                if phase == 0
                else float("inf")
                if phase == 1
                else -float("inf"),
            )
            legacy = x.clone()
            legacy.masked_fill_(~mask[:, :, None], 0)
            safe_moe_sum(gems, legacy, ref)
            masked_moe_sum(gems, x, mask, actual)
            torch.testing.assert_close(actual, ref, rtol=0, atol=0, equal_nan=True)
        if m:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                masked_moe_sum(gems, x, mask, actual)
            for phase in [0, 1, 2]:
                x.copy_(original + 0.25 * phase)
                mask.copy_(
                    (torch.arange(m * k, device="cuda").reshape(m, k) + phase) % 4 != 0
                )
                x.masked_fill_(~mask[:, :, None], float("nan"))
                graph.replay()
                legacy = x.clone()
                legacy.masked_fill_(~mask[:, :, None], 0)
                safe_moe_sum(gems, legacy, ref)
                torch.testing.assert_close(actual, ref, rtol=0, atol=0, equal_nan=True)
            mask.zero_()
            x.fill_(float("nan"))
            graph.replay()
            assert torch.count_nonzero(actual).item() == 0
        records.append(
            {
                "dtype": str(dtype),
                "shape": [m, k, h],
                "numerical_phases": 3,
                "changed_mask_graph_phases": 4 if m else 0,
                "status": "passed",
            }
        )
        print(json.dumps(records[-1]), flush=True)
print(
    json.dumps({"status": "passed", "cases": len(records), "records": records}),
    flush=True,
)
