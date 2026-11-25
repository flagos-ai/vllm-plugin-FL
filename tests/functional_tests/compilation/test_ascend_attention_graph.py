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
_TOTAL_HEADS = 10
_HEAD_DIM = 128
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

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size, disable_custom_all_reduce=True
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(world_size, backend="hccl")
        try:
            yield get_tp_group()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _cache_metadata(
    bucket, real_requests, step, block_size, *, disjoint=False, compact=False
):
    max_pages = (8192 if compact else _MAX_LENGTH) // block_size
    lengths = torch.zeros(bucket, dtype=torch.int32)
    prefixes = torch.zeros_like(lengths)
    table = torch.zeros((bucket, max_pages), dtype=torch.int32)
    slots = torch.full((bucket,), -1, dtype=torch.int64)
    sequence_lengths = (1, 128, 129, 259, 513, 32768)
    prefix = 129 + step % 31
    if disjoint:
        prefix = 128
        sequence_lengths = (127, 128, 129, 255, 256, 257, 1152)
        history = 1 + torch.randperm(
            max_pages - 1, generator=torch.Generator().manual_seed(step)
        )
        history_start = 0
    for row in range(real_requests):
        length = sequence_lengths[(step + row) % len(sequence_lengths)]
        if compact:
            length = 6999 if step < 3 and row == (step - 1) % bucket else 6011
        lengths[row] = length
        prefixes[row] = 6010 if compact else (min(prefix, length) if row % 3 else 0)
        page_count = (length + block_size - 1) // block_size
        past_pages = page_count - 1
        if disjoint:
            table[row, :past_pages] = history[
                history_start : history_start + past_pages
            ]
            history_start += past_pages
        else:
            shared = torch.arange(1, max_pages, dtype=torch.int32).roll(step + row)
            table[row, :past_pages] = shared[:past_pages]
        final_page = max_pages + (row + step) % bucket
        table[row, page_count - 1] = final_page
        slots[row] = final_page * block_size + (length - 1) % block_size
    return lengths, prefixes, table, slots


def _reference(
    query, updates, initial_cache, lengths, prefixes, paged, slots, window, block_size
):
    """Apply token writes on CPU, then gather the permitted keys by position."""
    key_cache, value_cache = (tensor.clone() for tensor in initial_cache)
    keys, values = updates
    expected = torch.zeros_like(query, dtype=torch.float32)
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, block_size)
        key_cache[page, offset] = keys[row]
        value_cache[page, offset] = values[row]
    for row, length in enumerate(lengths.tolist()):
        if not length:
            continue
        tail = max(length - window, 0)
        positions = torch.cat(
            (torch.arange(min(int(prefixes[row]), tail)), torch.arange(tail, length))
        )
        pages = paged[row, positions // block_size]
        offsets = positions % block_size
        key = key_cache[pages, offsets].float()
        value = value_cache[pages, offsets].float()
        logits = torch.einsum("hd,thd->ht", query[row].float(), key) * _HEAD_DIM**-0.5
        expected[row] = torch.einsum("ht,thd->hd", logits.softmax(-1), value)
    return expected, (key_cache, value_cache)


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") not in {"1", "2"}
    or not {"RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}.issubset(os.environ),
    reason="Requires one or two NPU processes launched with torchrun",
)
@pytest.mark.parametrize(
    ("block_size", "compact"),
    [(128, False), (256, False), (256, True)],
    ids=["128", "256", "skip-pages"],
)
@torch.inference_mode()
def test_ocr_full_decode_updates_paged_rswa_graph(
    tp_group, monkeypatch, block_size, compact
):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.forward_context import (
        BatchDescriptor,
        ForwardContext,
        override_forward_context,
    )
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackend,
        AscendAttentionState,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.rswa_graph import RswaGraphRuntime

    device = torch.device("npu", torch.npu.current_device())
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size
    head_major = world_size == 1
    if compact and not head_major:
        pytest.skip("OCR skip-pages requires TP1")
    sizes = [2] if compact else _CAPTURE_SIZES
    max_length = 8192 if compact else _MAX_LENGTH
    local_heads = 2 if compact else _TOTAL_HEADS // world_size
    window = 128 if head_major else 64
    layer_name = "model.layers.0.self_attn.attn"
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_copy_inputs=False,
            cudagraph_capture_sizes=sizes,
            static_forward_context={},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="unlimited-ocr",
                architectures=["UnlimitedOCRForCausalLM"],
                rswa_window=window,
            ),
            max_model_len=max_length,
            dtype=_DTYPE,
            runner_type="generate",
            use_mla=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max(sizes), enable_chunked_prefill=True
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=block_size),
        additional_config={"ocr_tp1_optimized": head_major},
        speculative_config=None,
        quant_config=None,
    )
    backend = AscendAttentionBackend
    with set_current_vllm_config(config):
        impl = backend.get_impl_cls()(
            local_heads,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            local_heads,
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
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=local_heads,
        head_size=_HEAD_DIM,
        dtype=_DTYPE,
    )
    builder = backend.get_builder_cls()(spec, [layer_name], config, device)
    runtime = RswaGraphRuntime(vllm_config=config, full_graph_wrapper_type=GraphWrapper)
    assert runtime.reserve_graph_memory(device) > 0
    runtime.bind_attention_groups([[SimpleNamespace(backend=backend)]])

    pages = max_length // block_size + max(sizes)
    page_shape = (
        (pages, local_heads, block_size, _HEAD_DIM)
        if head_major
        else (pages, block_size, local_heads, _HEAD_DIM)
    )
    initial_banks = tuple(
        torch.randn(
            page_shape,
            generator=torch.Generator().manual_seed(181 + 2 * rank + i),
            dtype=_DTYPE,
        )
        for i in range(2)
    )
    if head_major:
        raw = torch.empty(pages * spec.page_size_bytes, dtype=torch.int8, device=device)
        logical_shape = backend.get_kv_cache_shape(
            pages, block_size, local_heads, _HEAD_DIM
        )
        with set_current_vllm_config(config):
            order = backend.get_kv_cache_stride_order()
        kv_cache = _reshape_attention_kv_cache(
            raw, spec, logical_shape, order, pages, None
        )
        assert kv_cache.shape == (2, pages, block_size, local_heads, _HEAD_DIM)
        assert kv_cache.data_ptr() == raw.data_ptr() and kv_cache.nbytes == raw.nbytes
        physical_banks = raw.view(_DTYPE).view(2, *page_shape)
        initial_cache = tuple(bank.transpose(1, 2) for bank in initial_banks)
        for i, view in enumerate(kv_cache):
            bank = view.transpose(1, 2)
            assert bank.shape == page_shape and bank.is_contiguous()
            assert bank.data_ptr() == physical_banks[i].data_ptr()
            physical_banks[i].copy_(initial_banks[i])
    else:
        initial_cache = initial_banks
        kv_cache = physical_banks = tuple(bank.to(device) for bank in initial_banks)
    layer.kv_cache = kv_cache
    outputs = {
        size: torch.empty((size, local_heads, _HEAD_DIM), device=device, dtype=_DTYPE)
        for size in sizes
    }
    slot_mappings = {
        size: torch.empty(size, device=device, dtype=torch.int64) for size in sizes
    }

    def forward(query, key, value):
        from vllm.forward_context import get_forward_context

        live = get_forward_context().attn_metadata[layer_name]
        output = impl.forward(
            layer, query, key, value, layer.kv_cache, live, outputs[query.shape[0]]
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
        for bucket in sizes
    }
    results = {}
    try:
        set_cudagraph_capturing_enabled(True)
        try:
            for bucket in reversed(sizes):
                lengths = torch.ones(bucket, dtype=torch.int32)
                prefixes = torch.zeros_like(lengths)
                table = torch.zeros(
                    (bucket, max_length // block_size), dtype=torch.int32
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

        assert len(wrapper.concrete_graph_entries) == len(sizes)
        if head_major:
            # Capture ran entirely with negative slots. Both banks must still
            # match all the original bytes, including page zero.
            for bank, original in zip(physical_banks, initial_banks):
                torch.testing.assert_close(bank.cpu(), original, rtol=0, atol=0)
            assert runtime._params is not None
            assert set(runtime._params.attn_params) == set(sizes)
            for (param,) in runtime._params.attn_params.values():
                assert param.ocr_kv_pool is layer.kv_cache
                for view, bank in zip((param.key, param.value), physical_banks):
                    assert view.is_contiguous() and view.shape == bank.shape
                    assert view.data_ptr() == bank.data_ptr()
        cases = [
            (bucket, max(1, bucket - step % 4), False)
            for step, bucket in enumerate([*sizes, 4, 64, 1], start=1)
        ]
        if compact:
            assert not runtime._params.skip_pages_logged
            cases = [(2, 2, False), (2, 2, False), (2, 1, False)]
        elif head_major:
            cases.extend((64, live, True) for live in (64, 44, 43))
        for step, (bucket, real_requests, disjoint) in enumerate(cases, start=1):
            lengths, prefixes, table, slots = _cache_metadata(
                bucket,
                real_requests,
                step,
                block_size,
                disjoint=disjoint,
                compact=compact,
            )
            live = metadata(bucket, real_requests, lengths, prefixes, table, slots)
            generator = torch.Generator().manual_seed(226 + step + rank * 31)
            fresh = tuple(
                torch.randn(buffer.shape, generator=generator, dtype=_DTYPE)
                for buffer in inputs[bucket]
            )
            for buffer, value in zip(inputs[bucket], fresh):
                buffer.copy_(value)
            for cache, original in zip(physical_banks, initial_banks):
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
                if head_major:
                    with monkeypatch.context() as patch:
                        patch.setattr(layer, "kv_cache", kv_cache.view_as(kv_cache))
                        with pytest.raises(RuntimeError, match="KV pool was rebound"):
                            run(live, inputs[bucket])
                    assert torch.isnan(outputs[bucket]).all().item()

            expected, expected_cache = _reference(
                fresh[0],
                fresh[1:],
                initial_cache,
                lengths,
                prefixes,
                table,
                slots,
                window,
                block_size,
            )
            # TP2 rounds to BF16 before HCCL. Its reference uses the CPU group.
            expected = expected.to(_DTYPE).float()
            if world_size > 1:
                torch.distributed.all_reduce(expected, group=tp_group.cpu_group)
            result = run(live, inputs[bucket])
            if compact:
                assert runtime._params.skip_pages_logged
                (param,) = runtime._params.attn_params[bucket]
                expected_table = table.clone()
                if step < 3:
                    row = step - 1
                    expected_table[row, 24:26] = table[row, 26:28]
                torch.testing.assert_close(param.block_table.cpu(), expected_table)

            torch.npu.synchronize()
            torch.testing.assert_close(
                result[:real_requests].cpu(),
                expected[:real_requests].to(_DTYPE),
                rtol=0.02,
                atol=0.02,
            )
            for cache, reference in zip(physical_banks, expected_cache):
                if head_major:
                    reference = reference.transpose(1, 2)
                torch.testing.assert_close(cache.cpu(), reference, rtol=0, atol=0)
            if real_requests < bucket:
                assert live.attn_mask[real_requests:].all().item()
    finally:
        runtime.clear_decoder_graphs()
