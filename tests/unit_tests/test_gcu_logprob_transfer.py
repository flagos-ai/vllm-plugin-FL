# Copyright (c) 2026 BAAI. All rights reserved.
"""Check the private logprob shim without requiring GCU hardware."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


def load_shim():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/gcu/impl/logprob.py"
    )
    spec = importlib.util.spec_from_file_location("gcu_logprob_transfer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_gcu_topk_is_unchanged():
    module = load_shim()
    x = torch.tensor([[3.0, 1.0, 5.0, 2.0]])
    actual = module._LogprobTorch.topk(x, 2)
    expected = torch.topk(x, 2)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.indices, expected.indices)


def test_gcu_transfers_only_indices(monkeypatch):
    module = load_shim()
    host_input = object()
    device = SimpleNamespace(type="gcu")
    input_ = Mock(device=device)
    input_.detach.return_value.cpu.return_value = host_input
    values = Mock()
    indices = Mock()
    device_indices = object()
    indices.to.return_value = device_indices
    topk = Mock(return_value=SimpleNamespace(values=values, indices=indices))
    monkeypatch.setattr(
        module,
        "torch",
        SimpleNamespace(
            topk=topk, return_types=SimpleNamespace(topk=lambda pair: pair)
        ),
    )
    result = module._LogprobTorch.topk(input_, 7, sorted=False)
    topk.assert_called_once_with(host_input, 7, sorted=False)
    values.to.assert_not_called()
    indices.to.assert_called_once_with(device)
    assert result == (values, device_indices)
