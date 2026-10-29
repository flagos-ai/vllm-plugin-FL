# SPDX-License-Identifier: Apache-2.0
"""Native-kernel launch presets: numerical and changed-input graph contracts."""

import os

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip("requires an assigned GPU", allow_module_level=True)

import flag_gems
import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next.launch_profile import (
    install_qwen38_launch_profile,
    qsa_topk,
)


@pytest.mark.parametrize(
    "rows,columns", [(1, 512), (8, 2048), (64, 8192), (256, 17600)]
)
def test_qsa_two_stage_profile_changed_graph(rows, columns):
    install_qwen38_launch_profile()
    torch.manual_seed(rows + columns)
    scores = torch.randn(rows, columns, device="cuda")
    qsa_topk(scores, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indices = qsa_topk(scores, 512)
    for _ in range(2):
        scores.copy_(torch.randn_like(scores))
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.topk(scores.cpu(), 512, dim=-1, sorted=True).values
        torch.testing.assert_close(
            values.cpu().sort(-1, descending=True).values, reference, atol=0, rtol=0
        )
        assert bool(((indices >= 0) & (indices < columns)).all())
        torch.testing.assert_close(
            torch.gather(scores, -1, indices), values, atol=0, rtol=0
        )


@pytest.mark.parametrize("rows", [1, 8, 64, 1024])
def test_mean_launch_presets_preserve_reference(rows):
    install_qwen38_launch_profile()
    torch.manual_seed(rows)
    for width in (128, 256, 2560):
        x = torch.randn(rows, width, device="cuda", dtype=torch.float32)
        actual = flag_gems.mean_dim(x, [-1])
        expected = x.cpu().mean(-1)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    x = torch.randn(rows, 4, 2560, device="cuda", dtype=torch.bfloat16)
    actual = flag_gems.mean_dim(x, [1])
    expected = x.cpu().float().mean(1).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
