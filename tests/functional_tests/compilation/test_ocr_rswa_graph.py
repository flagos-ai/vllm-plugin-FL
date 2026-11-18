# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise R-SWA graph capture and replay with two Ascend ranks."""

import os
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Requires two NPU processes launched with torchrun",
)

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
    config = NS(
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_copy_inputs=False,
            cudagraph_capture_sizes=_CAPTURE_SIZES,
            static_forward_context={},
        ),
        parallel_config=NS(
            tensor_parallel_size=2,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=NS(
            hf_config=NS(
                model_type="unlimited-ocr",
                architectures=["UnlimitedOCRForCausalLM"],
                rswa_window=_WINDOW,
            ),
            max_model_len=_MAX_LENGTH,
            dtype=_DTYPE,
            runner_type="generate",
            use_mla=False,
        ),
        scheduler_config=NS(
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
    layer = NS(layer_name=layer_name, impl=impl, _k_scale_float=1.0, _v_scale_float=1.0)
    config.compilation_config.static_forward_context[layer_name] = layer
    builder = backend.get_builder_cls()(None, [layer_name], config, device)
    runtime = GraphRuntimeController(
        vllm_config=config, full_graph_wrapper_type=GraphWrapper
    )
    assert runtime.prepare_rswa_graph(device) > 0
    runtime.bind_attention_groups([[NS(backend=backend)]])

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
        common = NS(
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
