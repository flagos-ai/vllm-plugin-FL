# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.worker import kimi_k3 as runner_mod


def test_reload_is_rejected_before_the_parent_mutates_parameters(monkeypatch):
    runner = runner_mod.K3ModelRunner.__new__(runner_mod.K3ModelRunner)
    runner.model = SimpleNamespace(weight=torch.arange(4))
    original = runner.model.weight.clone()
    parent = Mock(side_effect=lambda *a, **k: runner.model.weight.zero_())
    monkeypatch.setattr(runner_mod.GPUModelRunner, "reload_weights", parent)
    with pytest.raises(NotImplementedError, match="restart the worker"):
        runner.reload_weights()
    parent.assert_not_called()
    assert torch.equal(runner.model.weight, original)


def test_weight_transfer_is_rejected_before_runner_allocation(monkeypatch):
    parent = Mock()
    monkeypatch.setattr(runner_mod.GPUModelRunner, "__init__", parent)
    monkeypatch.setattr(
        runner_mod,
        "execution_config",
        lambda _: SimpleNamespace(enabled=True, host_pp_feedback=True),
    )
    cfg = SimpleNamespace(weight_transfer_config=object())
    with pytest.raises(NotImplementedError, match="weight updates"):
        runner_mod.K3ModelRunner(cfg, torch.device("cpu"), diagnostics=Mock())
    parent.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires device Graph replay"
)
def test_rejected_reload_preserves_captured_storage():
    runner = runner_mod.K3ModelRunner.__new__(runner_mod.K3ModelRunner)
    weight = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    runner.model = SimpleNamespace(weight=weight)
    inputs = torch.empty_like(weight)
    out = torch.empty_like(weight)
    inputs.fill_(1)
    torch.mul(inputs, weight, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mul(inputs, weight, out=out)

    with pytest.raises(NotImplementedError, match="restart the worker"):
        runner.reload_weights()
    for value in (2, -1, 3):
        inputs.fill_(value)
        graph.replay()
        expected = torch.arange(16, dtype=weight.dtype) * value
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
