# SPDX-License-Identifier: Apache-2.0
"""Native top-k selection, out strides and graph replay under FlagTree AABS."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton.knobs

from vllm.models.minimax_m3.common.ops import index_topk as native

spec = importlib.util.spec_from_file_location(
    "isolated_minimax_bounds",
    Path(__file__).resolve().parents[2] / "vllm_fl/patches/minimax_topk_bounds.py",
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def _reference(score, prefix, topk, init=0, local=1):
    heads, queries, _ = score.shape
    result = torch.full(
        (heads, queries, topk), -1, device=score.device, dtype=torch.int32
    )
    for q in range(queries):
        visible = (prefix + q + 128) // 128
        s = torch.nan_to_num(score[:, q, :visible].clone(), nan=-1e30)
        s[:, : min(init, visible)] = 1e30
        s[:, max(0, visible - local) :] = 1e29
        count = min(topk, visible)
        result[:, q, :count] = s.topk(count, dim=-1).indices.to(torch.int32)
    return result


@pytest.fixture(scope="module", autouse=True)
def install():
    assert triton.knobs.autotuning.adjust_block_size
    assert patch.patch_minimax_topk_bounds(
        SimpleNamespace(architecture="MiniMaxM3SparseForConditionalGeneration")
    )


@pytest.mark.parametrize(
    "queries,prefix,topk,width",
    [
        (1, 0, 16, 16),
        (4, 1024, 16, 16),
        (16, 1792, 16, 16),
        (4, 1024, 64, 16),
        (4, 3072, 16, 32),
    ],
)
def test_native_selection(queries, prefix, topk, width):
    torch.manual_seed(713)
    score = torch.randn((2, queries, width), device="cuda")
    cu = torch.tensor([0, queries], dtype=torch.int32, device="cuda")
    context = torch.tensor([prefix], dtype=torch.int32, device="cuda")
    # A transposed persistent buffer exercises native out-view strides.
    backing = torch.empty((queries + 4, 2, topk), device="cuda", dtype=torch.int32)
    out = backing.transpose(0, 1)
    result = native.minimax_m3_index_topk(
        score, cu, context, queries, topk, 0, 1, out=out
    )
    torch.testing.assert_close(result, _reference(score, prefix, topk), rtol=0, atol=0)
    assert result.data_ptr() == out.data_ptr()


def test_changed_input_graph():
    score = torch.randn((2, 4, 16), device="cuda")
    cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    context = torch.tensor([1024], dtype=torch.int32, device="cuda")
    out = torch.empty((2, 4, 16), device="cuda", dtype=torch.int32)

    def run():
        return native.minimax_m3_index_topk(score, cu, context, 4, 16, 0, 1, out=out)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for prefix in (0, 1024, 1792):
        score.copy_(torch.randn_like(score))
        context.fill_(prefix)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, _reference(score, prefix, 16), rtol=0, atol=0)
