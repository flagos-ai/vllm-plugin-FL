# SPDX-License-Identifier: Apache-2.0
"""Ascend attention graph capture and replay without model weights."""

import os
from dataclasses import replace
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


_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64]
_MAX_LENGTH = 32768
_BLOCK_SIZE = 128
_LOCAL_HEADS = 5
_HEAD_DIM = 128
_WINDOW = 64
_DTYPE = torch.bfloat16


@pytest.fixture(scope="module")
def tp_group():
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2, disable_custom_all_reduce=True
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=int(os.environ["RANK"]),
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(2, backend="hccl")
        try:
            yield get_tp_group()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _cache_metadata(bucket, real_requests, step):
    max_pages = _MAX_LENGTH // _BLOCK_SIZE
    lengths = torch.zeros(bucket, dtype=torch.int32)
    prefixes = torch.zeros_like(lengths)
    table = torch.zeros((bucket, max_pages), dtype=torch.int32)
    slots = torch.full((bucket,), -1, dtype=torch.int64)
    sequence_lengths = (1, 128, 129, 259, 513, 32768)
    for row in range(real_requests):
        length = sequence_lengths[(step + row) % len(sequence_lengths)]
        lengths[row] = length
        prefixes[row] = min(129 + step % 31, length) if row % 3 else 0
        page_count = (length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        shared = torch.arange(1, max_pages, dtype=torch.int32).roll(step + row)
        table[row, : page_count - 1] = shared[: page_count - 1]
        final_page = max_pages + (row + step) % bucket
        table[row, page_count - 1] = final_page
        slots[row] = final_page * _BLOCK_SIZE + (length - 1) % _BLOCK_SIZE
    return lengths, prefixes, table, slots


def _reference(query, updates, initial_cache, lengths, prefixes, paged, slots):
    """Apply token writes on CPU, then gather the permitted keys by position."""
    key_cache, value_cache = (tensor.clone() for tensor in initial_cache)
    keys, values = updates
    expected = torch.zeros_like(query, dtype=torch.float32)
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, _BLOCK_SIZE)
        key_cache[page, offset] = keys[row]
        value_cache[page, offset] = values[row]
    for row, length in enumerate(lengths.tolist()):
        if not length:
            continue
        tail = max(length - _WINDOW, 0)
        positions = torch.cat(
            (torch.arange(min(int(prefixes[row]), tail)), torch.arange(tail, length))
        )
        pages = paged[row, positions // _BLOCK_SIZE]
        offsets = positions % _BLOCK_SIZE
        key = key_cache[pages, offsets].float()
        value = value_cache[pages, offsets].float()
        logits = torch.einsum("hd,thd->ht", query[row].float(), key) * _HEAD_DIM**-0.5
        expected[row] = torch.einsum("ht,thd->hd", logits.softmax(-1), value)
    return expected, (key_cache, value_cache)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Requires two NPU processes launched with torchrun",
)
@torch.inference_mode()
def test_ocr_full_decode_updates_paged_rswa_graph(tp_group):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.forward_context import (
        BatchDescriptor,
        ForwardContext,
        override_forward_context,
    )
    from vllm.v1.attention.backend import AttentionType

    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackend,
        AscendAttentionState,
    )

    device = torch.device("npu", torch.npu.current_device())
    rank = tp_group.rank_in_group
    layer_name = "model.layers.0.self_attn.attn"
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_copy_inputs=False,
            cudagraph_capture_sizes=_CAPTURE_SIZES,
            static_forward_context={},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="unlimited-ocr",
                architectures=["UnlimitedOCRForCausalLM"],
                rswa_window=_WINDOW,
            ),
            max_model_len=_MAX_LENGTH,
            dtype=_DTYPE,
            runner_type="generate",
            use_mla=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max(_CAPTURE_SIZES), enable_chunked_prefill=True
        ),
        additional_config={},
        speculative_config=None,
        quant_config=None,
    )
    backend = AscendAttentionBackend
    with set_current_vllm_config(config):
        impl = backend.get_impl_cls()(
            _LOCAL_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            _LOCAL_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
    layer = SimpleNamespace(
        layer_name=layer_name, impl=impl, _k_scale_float=1.0, _v_scale_float=1.0
    )
    config.compilation_config.static_forward_context[layer_name] = layer
    builder = backend.get_builder_cls()(None, [layer_name], config, device)
    runtime = GraphRuntimeController(
        vllm_config=config, full_graph_wrapper_type=GraphWrapper
    )
    assert runtime.prepare_rswa_graph(device) > 0
    runtime.bind_attention_groups([[SimpleNamespace(backend=backend)]])

    pages = _MAX_LENGTH // _BLOCK_SIZE + max(_CAPTURE_SIZES)
    initial_cache = tuple(
        torch.randn(
            (pages, _BLOCK_SIZE, _LOCAL_HEADS, _HEAD_DIM),
            generator=torch.Generator().manual_seed(181 + 2 * rank + i),
            dtype=_DTYPE,
        )
        for i in range(2)
    )
    kv_cache = tuple(tensor.to(device) for tensor in initial_cache)
    outputs = {
        size: torch.empty((size, _LOCAL_HEADS, _HEAD_DIM), device=device, dtype=_DTYPE)
        for size in _CAPTURE_SIZES
    }
    slot_mappings = {
        size: torch.empty(size, device=device, dtype=torch.int64)
        for size in _CAPTURE_SIZES
    }

    def forward(query, key, value):
        from vllm.forward_context import get_forward_context

        live = get_forward_context().attn_metadata[layer_name]
        output = impl.forward(
            layer, query, key, value, kv_cache, live, outputs[query.shape[0]]
        )
        return tp_group.all_reduce(output)

    def metadata(bucket, real_requests, lengths, prefixes, table, slots, capture=False):
        slot_mappings[bucket].copy_(slots)
        common = SimpleNamespace(
            query_start_loc_cpu=torch.arange(bucket + 1, dtype=torch.int32),
            num_reqs=bucket,
            num_actual_tokens=bucket,
            max_query_len=1,
            attn_state=AscendAttentionState.DecodeOnly,
            block_table_tensor=table.to(device),
            _seq_lens_cpu=lengths,
            slot_mapping=slot_mappings[bucket],
            rswa_prefix_lens=prefixes,
            rswa_num_actual_reqs=real_requests,
        )
        if capture:
            return builder.build_for_cudagraph_capture(common)
        return builder.build(0, common)

    def run(live, buffers, mode=CUDAGraphMode.FULL):
        context = ForwardContext(
            no_compile_layers=config.compilation_config.static_forward_context,
            attn_metadata={layer_name: live},
            slot_mapping={},
            cudagraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=buffers[0].shape[0], uniform=True
            ),
        )
        with override_forward_context(context):
            return wrapper(*buffers)

    wrapper = GraphWrapper(forward, config, CUDAGraphMode.FULL, graph_runtime=runtime)
    inputs = {
        bucket: tuple(torch.zeros_like(outputs[bucket]) for _ in range(3))
        for bucket in _CAPTURE_SIZES
    }
    results = {}
    try:
        set_cudagraph_capturing_enabled(True)
        try:
            for bucket in reversed(_CAPTURE_SIZES):
                lengths = torch.ones(bucket, dtype=torch.int32)
                prefixes = torch.zeros_like(lengths)
                table = torch.zeros(
                    (bucket, _MAX_LENGTH // _BLOCK_SIZE), dtype=torch.int32
                )
                slots = torch.full((bucket,), -1, dtype=torch.int64)
                live = metadata(
                    bucket, bucket, lengths, prefixes, table, slots, capture=True
                )
                # Match the runner's short, no-write warmup and then capture.
                run(live, inputs[bucket], CUDAGraphMode.NONE)
                results[bucket] = run(live, inputs[bucket])
        finally:
            set_cudagraph_capturing_enabled(False)

        assert len(wrapper.concrete_graph_entries) == len(_CAPTURE_SIZES)
        for step, bucket in enumerate([*_CAPTURE_SIZES, 4, 64, 1], start=1):
            real_requests = max(1, bucket - step % 4)
            lengths, prefixes, table, slots = _cache_metadata(
                bucket, real_requests, step
            )
            live = metadata(bucket, real_requests, lengths, prefixes, table, slots)
            generator = torch.Generator().manual_seed(226 + step + rank * 31)
            fresh = tuple(
                torch.randn(buffer.shape, generator=generator, dtype=_DTYPE)
                for buffer in inputs[bucket]
            )
            for buffer, value in zip(inputs[bucket], fresh):
                buffer.copy_(value)
            for cache, original in zip(kv_cache, initial_cache):
                cache.copy_(original)
            outputs[bucket].fill_(float("nan"))
            results[bucket].fill_(float("nan"))

            if step == 2:
                with pytest.raises(
                    RuntimeError, match="R-SWA graph block table exceeds"
                ):
                    run(
                        replace(live, block_tables=live.block_tables[:, :-1]),
                        inputs[bucket],
                    )
                assert torch.isnan(outputs[bucket]).all().item()

            expected, expected_cache = _reference(
                fresh[0], fresh[1:], initial_cache, lengths, prefixes, table, slots
            )
            # BF16 rounding precedes the collective, which uses HCCL in the graph
            # and the independent CPU process group for the expected result.
            expected = expected.to(_DTYPE).float()
            torch.distributed.all_reduce(expected, group=tp_group.cpu_group)
            result = run(live, inputs[bucket])
            torch.npu.synchronize()
            torch.testing.assert_close(
                result[:real_requests].cpu(),
                expected[:real_requests].to(_DTYPE),
                rtol=0.02,
                atol=0.02,
            )
            for cache, reference in zip(kv_cache, expected_cache):
                torch.testing.assert_close(cache.cpu(), reference, rtol=0, atol=0)
            if real_requests < bucket:
                assert live.attn_mask[real_requests:].all().item()
    finally:
        runtime.clear_decoder_graphs()
