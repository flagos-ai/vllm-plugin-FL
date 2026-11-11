# Copyright (c) 2026 BAAI. All rights reserved.
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

path = (
    Path(__file__).parents[2] / "vllm_fl/dispatch/backends/vendor/gcu/impl/logprob.py"
)
spec = importlib.util.spec_from_file_location("gcu_logprob_test_impl", path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


@pytest.mark.parametrize("size,k", [(33, 1), (4097, 5), (248320, 5)])
def test_topk_values_indices_and_dtype(size, k):
    device = "cpu"
    if os.environ.get("TEST_GCU_LOGPROB"):
        import torch_gcu  # noqa: F401

        device = "gcu"
    x = torch.arange(size, dtype=torch.float32).flip(0).reshape(1, -1)
    expected = torch.topk(x, k, dim=-1)
    actual = impl._LogprobTorch.topk(x.to(device), k, dim=-1)
    torch.testing.assert_close(actual.values.cpu(), expected.values, rtol=0, atol=0)
    torch.testing.assert_close(actual.indices.cpu(), expected.indices, rtol=0, atol=0)
    assert actual.indices.dtype == torch.int64
    assert actual.values.device.type == device
    assert impl._LogprobTorch().cat is torch.cat


def test_patch_preserves_signature_and_original_globals(monkeypatch):
    module = types.ModuleType("vllm.v1.worker.gpu.sample.logprob")
    sampler = types.ModuleType("vllm.v1.worker.gpu.sample.sampler")
    package = types.ModuleType("vllm.v1.worker.gpu.sample")
    package.logprob, package.sampler = module, sampler

    def original(logits, k=1, *, sorted=True):
        return torch.topk(logits, k, sorted=sorted)

    module.compute_topk_scores = sampler.compute_topk_scores = original
    for item in [package, module, sampler]:
        monkeypatch.setitem(sys.modules, item.__name__, item)
    assert impl.apply_logprob_gcu_patch()
    assert not impl.apply_logprob_gcu_patch()
    assert module.compute_topk_scores is sampler.compute_topk_scores
    assert original.__globals__["torch"] is torch
    actual = sampler.compute_topk_scores(torch.tensor([3.0, 1.0, 2.0]), sorted=False)
    assert actual.values.tolist() == [3.0]
    assert module.compute_topk_scores.__kwdefaults__ == {"sorted": True}
