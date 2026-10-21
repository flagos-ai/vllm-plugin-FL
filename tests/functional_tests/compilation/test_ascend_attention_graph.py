# SPDX-License-Identifier: Apache-2.0
"""Exercise PA/FIA task updates together, without model weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend graphs"
    ),
]


@pytest.mark.parametrize("order", [("pa", "fia"), ("fia", "pa"), ("pa", "pa")])
def test_attention_graph_updates_mixed_backends(order):
    import torch_npu

    from vllm_fl.compilation.graph_params import clear_graph_params, get_graph_params
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackendImpl,
        AscendAttentionState,
        AscendMetadata,
    )

    torch.npu.set_device(0)
    config = SimpleNamespace(
        speculative_config=None,
        quant_config=None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(sliding_window=64)),
    )
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1).npu()
    layers = []
    for index, backend in enumerate(order):
        impl = object.__new__(AscendAttentionBackendImpl)
        impl.vllm_config = config
        impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
        impl.scale, impl.sinks = 128**-0.5, None
        impl.sliding_window = 64 if backend == "fia" else None
        impl.key_cache = torch.randn(8, 128, 2, 128, dtype=torch.bfloat16, device="npu")
        impl.value_cache = torch.randn_like(impl.key_cache)
        query = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="npu")
        output = torch.empty_like(query)
        metadata = AscendMetadata(
            attn_mask=mask,
            attn_state=AscendAttentionState.DecodeOnly,
            num_actual_tokens=1,
            num_decode_tokens=1,
            num_decodes=1,
            block_tables=torch.arange(8, dtype=torch.int32).view(1, 8).npu(),
            # PA consumes the host lengths provided by the metadata builder.
            seq_lens=torch.tensor([64], dtype=torch.int32),
            seq_lens_list=[64],
            actual_seq_lengths_q=[1],
            causal=True,
        )
        layers.append((backend, impl, query, output, metadata, f"layer{index}"))

    def eager(backend, impl, query, metadata):
        expected = torch.empty_like(query)
        if backend == "pa":
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=impl.key_cache,
                value_cache=impl.value_cache,
                num_kv_heads=impl.num_kv_heads,
                num_heads=impl.num_heads,
                scale_value=impl.scale,
                block_table=metadata.block_tables,
                context_lens=metadata.seq_lens,
                out=expected,
            )
        else:
            key, value, block_size, table, lengths = impl._get_fia_params(
                None, None, metadata
            )
            args = dict(
                query=query,
                key=key,
                value=value,
                atten_mask=mask,
                block_table=table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=lengths,
                num_key_value_heads=impl.num_kv_heads,
                num_heads=impl.num_heads,
                scale=impl.scale,
                sparse_mode=4,
                pre_tokens=64,
                next_tokens=0,
            )
            expected = torch_npu.npu_fused_infer_attention_score(**args)[0]
        return expected

    for backend, impl, query, _, metadata, _ in layers:
        eager(backend, impl, query, metadata)
    torch.npu.synchronize()
    clear_graph_params()
    runtime = GraphRuntimeController(vllm_config=config)
    runtime.attention_impls = (AscendAttentionBackendImpl,)
    context = SimpleNamespace(
        batch_descriptor=SimpleNamespace(num_tokens=1),
        attn_metadata={name: metadata for _, _, _, _, metadata, name in layers},
        capturing=False,
    )
    graph = torch.npu.NPUGraph()
    with runtime.capture_scope(context), torch.npu.graph(graph):
        for backend, impl, query, output, metadata, name in layers:
            layer = SimpleNamespace(layer_name=name)
            if backend == "pa":
                impl._capture_paged_attention(query, metadata, output, layer)
            else:
                impl._capture_fused_infer_attention(
                    query, None, None, metadata, output, layer
                )

    assert (get_graph_params().workspaces[1] is not None) == ("fia" in order)
    for length in (65, 257, 129):
        expected = []
        for backend, impl, query, _, metadata, _ in layers:
            query.normal_()
            metadata.block_tables.copy_(metadata.block_tables.roll(1, dims=1))
            metadata.seq_lens.fill_(length)
            metadata.seq_lens_list = [length]
            expected.append(eager(backend, impl, query, metadata))
        with runtime.replay_scope(context):
            graph.replay()
        torch.npu.synchronize()
        for (_, _, _, output, _, _), reference in zip(layers, expected):
            torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
    clear_graph_params()
