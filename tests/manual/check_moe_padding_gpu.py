# SPDX-License-Identifier: Apache-2.0
"""Check real FlagGems expert padding and changed-input CUDA graph replay.

Run from the repository with PYTHONPATH=. on an otherwise available NVIDIA GPU:
    python tests/manual/check_moe_padding_gpu.py

This is an operator correctness check, not a throughput benchmark.
"""

import importlib
import json

import torch

from vllm_fl.ops.fused_moe.padding import (
    install_no_tle_moe_alignment,
    prepare_padded_moe_inputs,
    zero_padded_moe_rows,
)


def check_case(module, dtype, w1, w2, rows, nan_dummy):
    hidden = torch.randn((rows, 2048), device="cuda", dtype=dtype) * 0.2
    ids = torch.rand((rows, 256)).topk(8, dim=-1).indices.to("cuda", torch.int32)
    weights = torch.full((rows, 8), 0.125, device="cuda")
    mask = torch.arange(rows, device="cuda") % 3 == 1
    saved = None
    if nan_dummy:
        ids.copy_(torch.arange(1, 9, device="cuda", dtype=torch.int32).expand(rows, -1))
        saved = w1[0].clone()
        w1[0].fill_(float("nan"))
    try:
        reference = module.fused_experts_impl(
            hidden, w1, w2, weights, ids, inplace=False, activation="silu"
        )
        torch.cuda.synchronize()
        original = [x.clone() for x in (hidden, weights, ids)]

        def run():
            h, w, i = prepare_padded_moe_inputs(hidden, weights, ids, mask)
            result = module.fused_experts_impl(
                h, w1, w2, w, i, inplace=False, activation="silu"
            )
            return zero_padded_moe_rows(result, mask)

        def replace_inputs(new_mask):
            mask.copy_(new_mask)
            for tensor, baseline in zip((hidden, weights, ids), original):
                tensor.copy_(baseline)
            hidden.masked_fill_(mask[:, None], float("nan"))
            ids.masked_fill_(mask[:, None], -1)
            weights.masked_fill_(mask[:, None], float("inf"))

        def validate(result):
            torch.cuda.synchronize()
            torch.testing.assert_close(
                result[~mask], reference[~mask], rtol=0.03, atol=0.003
            )
            assert bool((result[mask] == 0).all()), "Nonfinite padding leaked"

        replace_inputs(torch.arange(rows, device="cuda") % 3 == 1)
        validate(run())
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        replace_inputs(torch.arange(rows, device="cuda") % 3 == 0)
        graph.replay()
        validate(result)
    finally:
        if saved is not None:
            w1[0].copy_(saved)
    print(
        json.dumps(
            {"dtype": str(dtype), "rows": rows, "nan_dummy": nan_dummy, "passed": True}
        ),
        flush=True,
    )


def main():
    import flag_gems

    assert torch.cuda.is_available(), "An available NVIDIA GPU is required"
    flag_gems.enable()
    install_no_tle_moe_alignment()
    module = importlib.import_module("flaggems_vllm.ops.fused_moe")
    align = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
    assert module.moe_align_block_size is align.moe_align_block_size_no_tle
    torch.manual_seed(334019)
    for dtype in (torch.bfloat16, torch.float16):
        w1 = torch.randn((256, 512, 2048), device="cuda", dtype=dtype) * 0.05
        w2 = torch.randn((256, 2048, 256), device="cuda", dtype=dtype) * 0.05
        for rows, nan_dummy in (
            (1, False),
            (8, False),
            (120, False),
            (128, False),
            (4096, False),
            (8, True),
        ):
            check_case(module, dtype, w1, w2, rows, nan_dummy)


if __name__ == "__main__":
    main()
