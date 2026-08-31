# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the DeepSeek-V4 attention compile boundary."""

from types import SimpleNamespace

import torch

from vllm_fl.models import deepseek_v4


def test_deepseek_v4_fl_attention_writes_preallocated_output(monkeypatch):
    calls = []

    class Layer:
        def attention_impl(self, *args):
            calls.append(args)
            args[-1].fill_(7)

    layer = Layer()
    monkeypatch.setattr(
        deepseek_v4,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer": layer}),
    )

    tensors = [torch.empty(1) for _ in range(7)]
    out = torch.empty(2, 3, 4)
    result = deepseek_v4._deepseek_v4_fl_attention(*tensors, out, "layer")

    assert result is None
    assert len(calls) == 1
    assert calls[0] == (*tensors, out)
    assert calls[0][-1] is out
    assert torch.equal(out, torch.full_like(out, 7))
    schema = torch._C._dispatch_find_schema_or_throw(
        "vllm::deepseek_v4_fl_attention", ""
    ).schema()
    assert "Tensor(a7!) out" in str(schema)
