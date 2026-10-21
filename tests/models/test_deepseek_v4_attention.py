# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the DeepSeek-V4 attention compile boundary."""

from types import SimpleNamespace

import torch

from vllm_fl.models import deepseek_v4


def test_deepseek_v4_fl_attention_workspace_uses_dedicated_workspace(
    monkeypatch,
):
    calls = []

    class Workspace:
        decode_capacity = 8

        def get(self, num_tokens, prefill_tail_bytes, prefill_gather_bytes):
            assert num_tokens == 2
            assert prefill_tail_bytes == 0
            assert prefill_gather_bytes == 0
            return (
                torch.empty(2, 3, 4),
                torch.empty(2, 3, 4),
                torch.empty(0, dtype=torch.uint8),
                torch.empty(0, dtype=torch.uint8),
            )

    class Layer:
        _fl_attention_io_workspace = Workspace()

        def _fl_prefill_workspace_bytes(self, num_tokens):
            assert num_tokens == 2
            return 0, 0

        def attention_impl(self, *args):
            calls.append(args)
            args[-4].fill_(7)

    layer = Layer()
    monkeypatch.setattr(
        deepseek_v4,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer": layer}),
    )

    tensors = [torch.empty(2) for _ in range(9)]
    result = deepseek_v4._deepseek_v4_fl_attention_workspace(*tensors, 3, 3, 4, "layer")

    assert result.shape == (2, 3, 4)
    assert torch.equal(result, torch.full_like(result, 7))
    assert len(calls) == 1
    assert calls[0][:-4] == tuple(tensors)
    assert calls[0][-4].data_ptr() == result.data_ptr()
    schema = torch._C._dispatch_find_schema_or_throw(
        "vllm::deepseek_v4_fl_attention_workspace", ""
    ).schema()
    assert "Tensor(a" not in str(schema)
    assert "-> Tensor" in str(schema)


def test_deepseek_v4_fl_attention_compile_op_declares_output_mutation(
    monkeypatch,
):
    calls = []

    class Workspace:
        decode_capacity = 8

        def get(self, num_tokens, prefill_tail_bytes, prefill_gather_bytes):
            assert (num_tokens, prefill_tail_bytes, prefill_gather_bytes) == (
                2,
                0,
                0,
            )
            return (
                torch.empty(2, 3, 4),
                torch.empty(2, 3, 4),
                torch.empty(0, dtype=torch.uint8),
                torch.empty(0, dtype=torch.uint8),
            )

    class Layer:
        _fl_attention_io_workspace = Workspace()

        def _fl_prefill_workspace_bytes(self, num_tokens):
            assert num_tokens == 2
            return 0, 0

        def attention_impl(self, *args):
            calls.append(args)
            args[-4].fill_(11)

    monkeypatch.setattr(
        deepseek_v4,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer": Layer()}),
    )

    tensors = [torch.empty(2) for _ in range(9)]
    out = torch.empty(2, 3, 4)
    result = deepseek_v4._deepseek_v4_fl_attention(*tensors, out, "layer")

    assert result is None
    assert torch.equal(out, torch.full_like(out, 11))
    assert len(calls) == 1
    assert calls[0][:-4] == tuple(tensors)
    assert calls[0][-4] is out
    schema = torch._C._dispatch_find_schema_or_throw(
        "vllm::deepseek_v4_fl_attention", ""
    ).schema()
    assert "!) out" in str(schema)
    assert "-> ()" in str(schema)


def test_attention_io_workspace_separates_decode_and_prefill(monkeypatch):
    class FakeManager:
        def __init__(self, device, num_ubatches):
            self.calls = []
            self.num_ubatches = num_ubatches

        def get_simultaneous(self, *specs):
            self.calls.append(specs)
            return [torch.empty(shape, dtype=dtype) for shape, dtype in specs]

    monkeypatch.setattr(deepseek_v4, "WorkspaceManager", FakeManager)
    workspace = deepseek_v4._DeepseekV4AttentionIOWorkspace(
        decode_capacity=8,
        padded_heads=3,
        head_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_ubatches=0,
    )

    q_decode, o_decode, decode_tail, decode_gather = workspace.get(2)
    prefill_shape = (12, 3, 4)
    output_bytes = deepseek_v4._workspace_nbytes(prefill_shape, torch.bfloat16)
    gather_bytes = 256
    q_prefill, o_prefill, prefill_tail, prefill_gather = workspace.get(
        12, output_bytes + gather_bytes, gather_bytes
    )

    assert q_decode.shape == o_decode.shape == (2, 3, 4)
    assert q_prefill.shape == o_prefill.shape == (12, 3, 4)
    assert decode_tail.numel() == 0
    assert decode_gather.numel() == 0
    assert prefill_tail.numel() == output_bytes + gather_bytes
    assert prefill_gather.numel() == gather_bytes
    assert (
        o_prefill.untyped_storage().data_ptr()
        == prefill_tail.untyped_storage().data_ptr()
    )
    assert prefill_gather.data_ptr() == prefill_tail.data_ptr() + output_bytes
    assert workspace.decode.num_ubatches == 1
    assert workspace.prefill.num_ubatches == 1
    assert workspace.decode.calls[0][0][0] == (8, 3, 4)
    assert workspace.prefill.calls[0][0][0] == (12, 3, 4)
    assert workspace.prefill.calls[0][1][0] == (output_bytes + 256,)
