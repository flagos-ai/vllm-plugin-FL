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


def _attention_layer(name, quant_config=None, kv_dtype="auto"):
    from vllm.model_executor.layers.attention.attention import (
        Attention,
        _init_kv_cache_quant,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsKVCacheMethod,
    )

    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = name
    layer.kv_cache_dtype = kv_dtype
    layer.quant_method = None
    _init_kv_cache_quant(layer, quant_config, name)
    if quant_config is not None:
        assert isinstance(layer.quant_method, CompressedTensorsKVCacheMethod)
        assert layer.quant_method.quant_config is quant_config
    return layer


@pytest.mark.parametrize("order", [("pa", "fia"), ("fia", "pa"), ("pa", "pa")])
@pytest.mark.parametrize("linear_quantized", [False, True])
def test_attention_graph_updates_mixed_backends(order, linear_quantized):
    import torch_npu

    from vllm_fl.compilation.graph_params import clear_graph_params, get_graph_params
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackendImpl,
        AscendAttentionState,
        AscendMetadata,
    )

    torch.npu.set_device(0)
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendCompressedTensorsConfig,
    )

    config = SimpleNamespace(
        speculative_config=None,
        quant_config=AscendCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="int-quantized"
        )
        if linear_quantized
        else None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(sliding_window=64)),
    )
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1).npu()
    layers = []
    for index, backend in enumerate(order):
        impl = object.__new__(AscendAttentionBackendImpl)
        impl.vllm_config = config
        impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
        impl.scale, impl.sinks = 128**-0.5, None
        impl.kv_cache_dtype = "auto"
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

    attention_layers = {
        name: _attention_layer(name, config.quant_config)
        for _, _, _, _, _, name in layers
    }
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
            layer = attention_layers[name]
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


@pytest.mark.parametrize(
    "method,kv_dtype,cache_dtype,rejected",
    [
        ("none", "auto", torch.bfloat16, False),
        ("unquantized", "float16", torch.float16, False),
        ("quantized", "auto", torch.bfloat16, True),
        ("ct_linear", "auto", torch.bfloat16, False),
        ("ct_linear", "float16", torch.float16, False),
        ("ct_fp8", "auto", torch.bfloat16, True),
        ("ct_linear", "auto", torch.int8, True),
        ("none", "int8", torch.bfloat16, True),
        ("none", "fp8", torch.bfloat16, True),
        ("none", "auto", torch.int8, True),
    ],
)
def test_attention_graph_quantization_gate(method, kv_dtype, cache_dtype, rejected):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackendImpl,
        AscendAttentionState,
    )
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendCompressedTensorsConfig,
    )

    quant_config = AscendCompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="int-quantized",
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
        }
        if method == "ct_fp8"
        else None,
    )
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.vllm_config = SimpleNamespace(
        speculative_config=None, quant_config=quant_config
    )
    impl.sinks = None
    impl.kv_cache_dtype = kv_dtype
    impl.key_cache = torch.empty(1, dtype=cache_dtype, device="npu")
    impl.value_cache = torch.empty_like(impl.key_cache)
    layer = _attention_layer(
        "layer", quant_config if method.startswith("ct_") else None, kv_dtype
    )
    if not method.startswith("ct_"):
        layer.quant_method = {
            "none": None,
            "unquantized": UnquantizedLinearMethod(),
            "quantized": object(),
        }[method]
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)
    if rejected:
        with pytest.raises(NotImplementedError, match="quantized attention"):
            impl._require_regular_graph_capture(metadata, layer)
    else:
        assert impl._require_regular_graph_capture(metadata, layer) == "layer"
