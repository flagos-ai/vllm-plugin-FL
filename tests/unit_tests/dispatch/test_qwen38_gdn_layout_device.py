# SPDX-License-Identifier: Apache-2.0
"""Real MetaX row-gather regression for mixed K1 decode plus two prefills."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1",
    reason="requires an assigned MetaX device and the pinned FlagGems runtime",
)


def test_strided_qkv_gather_eager_and_changed_input_graph_under_flaggems():
    import flag_gems

    from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax

    torch.manual_seed(942)
    projected = torch.randn(42, 2048, device="cuda", dtype=torch.bfloat16)
    changed = torch.randn_like(projected)
    qkv = projected[:, :1280]
    assert qkv.shape == (42, 1280) and qkv.stride() == (2048, 1)
    spec_indices = torch.arange(2, device="cuda", dtype=torch.int64)
    prefill_indices = torch.arange(2, 42, device="cuda", dtype=torch.int64)
    assert getattr(flag_gems, "current_work_registrar", None) is None
    # Native Torch references stay outside the FlagGems registration context.
    expected = tuple(
        qkv.index_select(0, index) for index in (spec_indices, prefill_indices)
    )
    expected_changed = tuple(
        changed[:, :1280].index_select(0, index)
        for index in (spec_indices, prefill_indices)
    )

    def gather():
        packed = metax.qwen38_gdn_mixed_qkv(qkv)
        assert packed.is_contiguous()
        return tuple(
            packed.index_select(0, index) for index in (spec_indices, prefill_indices)
        )

    with flag_gems.use_gems():
        assert getattr(flag_gems, "current_work_registrar", None) is not None
        eager = gather()
        for _ in range(2):
            gather()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gather()
        projected.copy_(changed)
        graph.replay()

    for actual, reference in zip(eager, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for actual, reference in zip(captured, expected_changed):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    print(
        {
            "shape": list(qkv.shape),
            "input_stride": list(qkv.stride()),
            "spec_tokens": 2,
            "prefill_tokens": 40,
            "eager_and_changed_graph_exact": True,
            "flag_gems_context": True,
        }
    )
