# Copyright (c) 2026 BAAI. All rights reserved.
"""Regression coverage for the scoped CPU top-k fallback and its consumer."""

import os

import pytest
import torch

if os.environ.get("FL_TEST_PLATFORM", "enflame") != "enflame":
    pytest.skip("GCU-specific tests", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import logprob as adapter


def test_cpu_topk_keeps_native_semantics():
    x = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
    result = adapter._LogprobTorch.topk(x, 2, largest=False)
    expected = torch.topk(x, 2, largest=False)
    torch.testing.assert_close(result.values, expected.values)
    assert torch.equal(result.indices, expected.indices)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "rows,width,k,largest",
    [(1, 128, 0, True), (1, 128, 5, False), (1, 151936, 5, True), (4, 248320, 5, True)],
)
def test_real_gcu_topk_fallback_and_consumer(monkeypatch, rows, width, k, largest):
    from vllm.platforms import current_platform

    if current_platform.device_type != "gcu":
        pytest.skip("Requires GCU")
    import flag_gems

    from vllm.v1.worker.gpu.sample import logprob

    def unexpected(*args, **kwargs):
        pytest.fail("Fallback must not enter the public top-k compiler")

    monkeypatch.setattr(flag_gems, "topk", unexpected)
    torch.manual_seed(182)
    cpu = torch.randn(rows, width)
    x = cpu.to("gcu")
    actual = adapter._LogprobTorch.topk(x, k, largest=largest)
    expected = torch.topk(cpu, k, largest=largest)
    assert actual.values.device.type == "cpu"
    assert actual.indices.device == x.device
    assert actual.indices.dtype == torch.int64
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    assert torch.equal(actual.indices.cpu(), expected.indices)
    ids = torch.arange(rows, device=x.device, dtype=torch.int64)
    adapter.apply_logprob_gcu_patch()
    with flag_gems.use_gems(include=["cumsum"]):
        result = logprob.compute_topk_scores(x, k, ids)
    # The consumer always requests largest=True, independently of the direct test.
    tokens = torch.cat((ids.cpu()[:, None], torch.topk(cpu, k).indices), -1)
    assert torch.equal(result.logprob_token_ids.cpu(), tokens)
    torch.testing.assert_close(
        result.logprobs.cpu(), cpu.log_softmax(-1).gather(-1, tokens)
    )
    ranks = (cpu >= cpu.gather(-1, ids.cpu()[:, None])).sum(-1)
    assert torch.equal(result.selected_token_ranks.cpu(), ranks)
