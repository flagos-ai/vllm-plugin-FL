# Copyright (c) 2026 BAAI. All rights reserved.
"""Real NPU task capture/update/replay against the native eager attention path."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.functional]


@pytest.fixture
def ascend_runtime(backend, request, monkeypatch):
    # Do not import Ascend modules (or request an NPU device) on CPU/CUDA hosts.
    if backend != "ascend":
        pytest.skip("Ascend NPU graph replay requires the Ascend backend")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    device = request.getfixturevalue("device")
    load_attention = request.getfixturevalue("ascend_attention_module")

    import vllm_fl.compilation.graph as graph_state

    monkeypatch.setattr(graph_state, "_ascend_graph_params", None)
    monkeypatch.setattr(graph_state, "_ascend_graph_capturing", False)
    try:
        attention = load_attention()
        with torch.npu.device(device):
            torch.npu.synchronize()
            yield attention, graph_state, device
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        graph_state.set_ascend_graph_capturing(False)
        graph_state._ascend_graph_params = None


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("windows", [(8,), (None, 8, 64)], ids=["fia", "pa_fia"])
def test_ascend_attention_graph_replay(ascend_runtime, batch, windows):
    attention, state, device = ascend_runtime
    impl_cls = attention.AscendAttentionBackendImpl
    state.set_ascend_graph_params([batch])
    # Keep all captured buffers alive until the final replay has synchronized.
    query = torch.randn(batch, 4, 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(4, 128, 2, 128, dtype=query.dtype, device=device)
    value = torch.randn_like(key)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1).to(device)
    layers = []
    for index, window in enumerate(windows):
        impl = object.__new__(impl_cls)
        impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
        impl.scale = 128**-0.5
        impl.sliding_window = window
        impl.key_cache, impl.value_cache = key, value
        layers.append((f"layer{index}", impl, torch.empty_like(query)))

    def metadata(lengths, swapped=False):
        return attention.AscendMetadata(
            attn_state=attention.AscendAttentionState.DecodeOnly,
            attn_mask=mask,
            num_actual_tokens=batch,
            actual_seq_lengths_q=list(range(1, batch + 1)),
            # PA requires CPU context lengths, even when query/cache are NPU tensors.
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
            seq_lens_list=list(lengths),
            block_tables=torch.tensor(
                [[2, 3] if swapped else [0, 1] for _ in lengths],
                dtype=torch.int32,
                device=device,
            ),
        )

    captured = metadata([3] * batch)
    for name, impl, output in layers:
        impl.forward_impl(query, key, value, (), captured, output, name)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    try:
        state.set_ascend_graph_capturing(True)
        try:
            with torch.npu.graph(graph, stream=stream):
                for name, impl, output in layers:
                    impl.forward_impl(query, key, value, (), captured, output, name)
        finally:
            state.set_ascend_graph_capturing(False)
        state.weak_ref_ascend_graph_workspaces()
        for step, length in enumerate((9, 129, 133)):
            live = metadata([length + i for i in range(batch)], swapped=bool(step % 2))
            expected = []
            for name, impl, output in layers:
                reference = torch.empty_like(output)
                impl.forward_impl(query, key, value, (), live, reference, name)
                expected.append(reference)
            torch.npu.synchronize()
            graph.replay()
            impl_cls.update_graph_params(
                update_stream,
                SimpleNamespace(attn_metadata={name: live for name, _, _ in layers}),
                batch,
            )
            torch.npu.synchronize()
            for (_, _, output), reference in zip(layers, expected):
                assert torch.isfinite(output).all().item()
                torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
    finally:
        torch.npu.synchronize()
        graph.reset()
