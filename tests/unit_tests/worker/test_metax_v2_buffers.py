# Copyright (c) 2026 BAAI. All rights reserved.
"""Real upstream V2 kernels on MetaX, without a model or process group.

Collection needs no vLLM installation. GPU execution is a separate validation:
these contracts do not establish model accuracy or distributed correctness.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.serial]


@pytest.fixture
def metax_v2_runtime(monkeypatch):
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax" or not torch.cuda.is_available():
        pytest.skip("Requires a real MetaX GPU")

    from vllm.platforms import current_platform
    from vllm.utils import torch_utils
    from vllm.v1.worker.gpu import buffer_utils

    from vllm_fl.worker.metax_v2 import prepare_metax_v2_runtime

    assert current_platform.vendor_name == "metax"
    assert current_platform.device_type == "cuda"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Register restoration before invoking the real production adapter. Do not
    # replace buffer classes, kernels, platform predicates or CUDA primitives.
    monkeypatch.setattr(buffer_utils, "is_uva_available", buffer_utils.is_uva_available)
    monkeypatch.setattr(torch_utils, "weak_ref_tensor", torch_utils.weak_ref_tensor)
    assert prepare_metax_v2_runtime(SimpleNamespace(use_v2_model_runner=True))
    try:
        native_view = torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::get_cuda_view_from_cpu_tensor", "CPU"
        )
    except RuntimeError:
        native_view = False
    if not native_view:
        assert not buffer_utils.is_uva_available()
    with torch.inference_mode():
        yield torch.device("cuda"), buffer_utils


def _exact(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_metax_v2_non_uva_buffer_dynamic_copy(metax_v2_runtime):
    _, buffers = metax_v2_runtime
    buffer = buffers.NonUvaBuffer((5, 3), torch.int64)
    assert buffer.cpu.device.type == "cpu"
    assert buffer._uva.device.type == "cuda"
    assert not buffer.cpu.is_pinned()
    pointer = buffer._uva.data_ptr()
    for shift in (0, 37, -91):
        expected = torch.arange(15).reshape(5, 3) + shift + 2**35
        buffer.np[:] = expected.numpy()
        _exact(buffer.uva(3), expected[:3])
        _exact(buffer.uva(), expected)
        assert buffer._uva.data_ptr() == pointer


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_metax_v2_pool_concurrent_slots_growth_and_values(metax_v2_runtime, dtype):
    _, buffers = metax_v2_runtime
    pool = buffers.UvaBufferPool((2, 3), dtype, max_concurrency=2)
    if not buffers.is_uva_available():
        assert all(isinstance(slot, buffers.NonUvaBuffer) for slot in pool._uva_bufs)

    first_cpu = torch.arange(6, dtype=dtype).reshape(2, 3) + 11
    second_cpu = first_cpu + 101
    first = pool.copy_to_uva(first_cpu)
    # Both slots remain live until their queued GPU readers have completed.
    first_read = first.clone()
    second = pool.copy_to_uva(second_cpu.numpy())
    second_read = second.clone()
    assert first.data_ptr() != second.data_ptr()
    _exact(first_read, first_cpu)
    _exact(second_read, second_cpu)
    _exact(first, first_cpu)

    # The next turn grows the first slot instead of overwriting its old storage.
    grown_cpu = torch.arange(15, dtype=dtype).reshape(5, 3) - 17
    grown = pool.copy_to_uva(grown_cpu.tolist())
    assert pool._uva_bufs[pool._curr].cpu.shape == (8, 3)
    assert grown.data_ptr() != first.data_ptr()
    _exact(grown, grown_cpu)
    _exact(first, first_cpu)
    _exact(second, second_cpu)

    # Reuse a retired slot with new values and exercise copy_to_gpu(out=...).
    output = torch.empty((2, 3), dtype=dtype, device="cuda")
    updated = -second_cpu
    result = pool.copy_to_gpu(updated, out=output)
    assert result.data_ptr() == output.data_ptr()
    assert result.data_ptr() != pool._uva_bufs[pool._curr]._uva.data_ptr()
    _exact(result, updated)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_metax_v2_staged_write_boundaries_and_reuse(metax_v2_runtime, dtype):
    device, buffers = metax_v2_runtime
    tensor = buffers.StagedWriteTensor(
        (4, 2053), dtype, device, max_concurrency=2, uva_instead_of_gpu=True
    )
    if not buffers.is_uva_available():
        assert tensor.write_contents is None
    sentinel = -777
    tensor.gpu.fill_(sentinel)
    expected = torch.full((4, 2053), sentinel, dtype=dtype)
    for round_index in range(3):
        bias = 2**35 if dtype == torch.int64 else 0
        updates = [(0, 1019, 17), (1, 2048, 5), (3, 0, 1031)]
        for row, start, length in updates:
            values = torch.arange(length, dtype=dtype) + bias + round_index * 31
            if dtype == torch.float32:
                values = values / 4
            tensor.stage_write(row, start, values.tolist())
            expected[row, start : start + length] = values
        tensor.stage_write(2, 0, [])
        tensor.apply_write()
        _exact(tensor.gpu, expected)
        assert not tensor._staged_write_indices
        tensor.apply_write()  # An empty batch must retain every element.
        _exact(tensor.gpu, expected)

    scalar = buffers.StagedWriteTensor(4, dtype, device, max_concurrency=2)
    scalar.gpu.fill_(sentinel)
    scalar.stage_write_elem(3, 123)
    scalar.apply_write()
    _exact(scalar.gpu, torch.tensor([sentinel, sentinel, sentinel, 123], dtype=dtype))


def test_metax_v2_fused_staged_writer_independent_groups(metax_v2_runtime):
    device, buffers = metax_v2_runtime
    tensors = [
        buffers.StagedWriteTensor((3, width), torch.int32, device, max_concurrency=2)
        for width in (2053, 1031)
    ]
    writer = buffers.FusedStagedWriter(device, max_writes=1, max_concurrency=2)
    pointers = torch.tensor(
        [tensor.gpu.data_ptr() for tensor in tensors], dtype=torch.uint64, device=device
    )
    strides = torch.tensor(
        [tensor.gpu.stride(0) for tensor in tensors], dtype=torch.int64, device=device
    )
    expected = [
        torch.full(tensor.gpu.shape, -971, dtype=torch.int32) for tensor in tensors
    ]
    for tensor in tensors:
        tensor.gpu.fill_(-971)
    for round_index in range(3):
        for group, tensor in enumerate(tensors):
            width = tensor.gpu.shape[1]
            for row, start, length in ((0, 1019, 12), (2, width - 5, 5)):
                values = (
                    torch.arange(length, dtype=torch.int32)
                    + 100 * group
                    + 17 * round_index
                )
                tensor.stage_write(row, start, values.tolist())
                expected[group][row, start : start + length] = values
        writer.apply(tensors, pointers, strides)
        for tensor, oracle in zip(tensors, expected):
            _exact(tensor.gpu, oracle)
            assert not tensor._staged_write_contents
        assert writer.indices._uva_bufs[writer.indices._curr].cpu.shape[0] == 4
        writer.apply(tensors, pointers, strides)
        for tensor, oracle in zip(tensors, expected):
            _exact(tensor.gpu, oracle)


@pytest.mark.parametrize("block_sizes", [(528,), (528, 128)])
def test_metax_v2_block_tables_slot_mapping_and_padding(metax_v2_runtime, block_sizes):
    from vllm.v1.worker.gpu.block_table import BlockTables

    device, _ = metax_v2_runtime
    tables = BlockTables(list(block_sizes), 3, 8, [6] * len(block_sizes), device)
    positions_cpu = [127, 527, 528, 15, 128]
    positions = torch.tensor(positions_cpu, dtype=torch.int64, device=device)
    query_start = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    for round_index in range(3):
        cpu_tables = {}
        for request in (0, 2):
            groups = tuple(
                [
                    11 + request * 19 + group * 101 + round_index * 41 + i * 3
                    for i in range(5)
                ]
                for group in range(len(block_sizes))
            )
            cpu_tables[request] = groups
            tables.append_block_ids(
                request, tuple(ids[:3] for ids in groups), overwrite=True
            )
            tables.append_block_ids(
                request, tuple(ids[3:] for ids in groups), overwrite=False
            )
        tables.apply_staged_writes()
        order = [2, 0] if round_index != 1 else [0, 2]
        mapping = torch.tensor(order, dtype=torch.int64, device=device)
        for group, actual in enumerate(
            tables.gather_block_tables(mapping, num_reqs_padded=3)
        ):
            expected = torch.zeros((3, 6), dtype=torch.int32)
            for row, request in enumerate(order):
                expected[row, :5] = torch.tensor(
                    cpu_tables[request][group], dtype=torch.int32
                )
            _exact(actual, expected)

        tables.slot_mappings.fill_(123456789)  # Detect stale slots in padded tokens.
        actual_slots = tables.compute_slot_mappings(mapping, query_start, positions, 8)
        expected_slots = torch.full((len(block_sizes), 8), -1, dtype=torch.int64)
        for group, block_size in enumerate(block_sizes):
            for token, position in enumerate(positions_cpu):
                request = order[0 if token < 3 else 1]
                block = cpu_tables[request][group][position // block_size]
                expected_slots[group, token] = (
                    block * block_size + position % block_size
                )
        _exact(actual_slots, expected_slots)


@pytest.mark.parametrize("vocab_size", [257, 248320])
def test_metax_v2_sampler_greedy_large_vocab(metax_v2_runtime, vocab_size):
    from vllm.config import CompilationConfig, VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device, _ = metax_v2_runtime
    assert Sampler.__module__ == "vllm.v1.worker.gpu.sample.sampler"
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    states = RequestState(2, 16, 4, 0, vocab_size, device)
    sampler = Sampler(config, 2, vocab_size, device, states)
    for request in range(2):
        req_id = f"v2-buffer-sampling-{request}"
        states.add_request(req_id, 1, [5 + request], 0, 8)
        sampler.add_request(
            states.req_id_to_index[req_id], SamplingParams(temperature=0, max_tokens=8)
        )
    states.apply_staged_writes()
    sampler.apply_staged_writes()
    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, device), is_padding=False)
    batch.req_ids = [states.index_to_req_id[index] for index in range(2)]
    for round_index in range(3):
        host_logits = torch.arange(vocab_size).remainder(37).float().repeat(2, 1)
        winners = [vocab_size - 1 - round_index, vocab_size // 2 + round_index]
        host_logits[0, winners[0]] = 1000
        host_logits[1, winners[1]] = 1001
        expected = host_logits.argmax(dim=-1, keepdim=True)
        logits = host_logits.to(device)
        output = sampler(logits, batch)
        _exact(output.sampled_token_ids, expected)
        assert output.num_sampled.cpu().tolist() == [1, 1]
        assert output.num_rejected.cpu().tolist() == [0, 0]
        assert output.logprobs_tensors is None
        _exact(logits, host_logits)


@contextmanager
def _capture_enabled():
    from vllm.compilation import monitor

    previous = monitor.cudagraph_capturing_enabled
    monitor.set_cudagraph_capturing_enabled(True)
    try:
        yield
    finally:
        monitor.set_cudagraph_capturing_enabled(previous)


def _call_native_graph(wrapper, value, config, descriptor):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    with set_forward_context(
        attn_metadata=None,
        vllm_config=config,
        num_tokens=value.shape[0],
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
    ):
        return wrapper(value)


@pytest.mark.parametrize(("dtype", "rows"), [(torch.float32, 2), (torch.bfloat16, 4)])
def test_metax_v2_native_graph_nondefault_stream_values(metax_v2_runtime, dtype, rows):
    from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationMode
    from vllm.forward_context import BatchDescriptor
    from vllm.platforms import current_platform

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[rows],
        )
    )
    assert config.use_v2_model_runner
    assert config.compilation_config.mode == CompilationMode.NONE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    stream.wait_stream(torch.cuda.current_stream())
    calls = []

    def runnable(value):
        calls.append(torch.cuda.current_stream().cuda_stream)
        return torch.relu(value * 2 + 1)

    descriptor = BatchDescriptor(num_tokens=rows, num_reqs=rows, uniform=True)
    from vllm.utils.import_utils import resolve_obj_by_qualname

    wrapper_factory = resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls()
    )
    wrapper = wrapper_factory(
        runnable,
        config,
        CUDAGraphMode.FULL,
        CUDAGraphOptions(debug_log_enable=False, weak_ref_output=True),
    )
    assert type(wrapper) is CUDAGraphWrapper
    try:
        with (
            torch.cuda.stream(stream),
            set_current_vllm_config(config),
            _capture_enabled(),
        ):
            host_input = (
                torch.arange(rows * 8).reshape(rows, 8).float().div(8).to(dtype)
            )
            static_input = host_input.to("cuda")
            for _ in range(3):
                runnable(static_input)
            stream.synchronize()
            calls.clear()
            captured_output = _call_native_graph(
                wrapper, static_input, config, descriptor
            )
            stream.synchronize()
            entry = wrapper.concrete_cudagraph_entries[descriptor]
            assert isinstance(entry.cudagraph, torch.cuda.CUDAGraph)
            assert calls == [stream.cuda_stream]
            captured_graph = entry.cudagraph
            for shift in (-3, 5, 11):
                host_input = (
                    (torch.arange(rows * 8).reshape(rows, 8).float() + shift) / 8
                ).to(dtype)
                static_input.copy_(host_input)
                captured_output.fill_(float("nan"))
                actual = _call_native_graph(wrapper, static_input, config, descriptor)
                stream.synchronize()
                expected = torch.relu(host_input.float() * 2 + 1).to(dtype)
                _exact(actual, expected)
                _exact(static_input, host_input)
                assert actual.data_ptr() == captured_output.data_ptr()
                assert (
                    wrapper.concrete_cudagraph_entries[descriptor].cudagraph
                    is captured_graph
                )
                assert calls == [
                    stream.cuda_stream
                ]  # Native replay, not Python fallback.
    finally:
        stream.synchronize()
        wrapper.clear_graphs()
        torch.cuda.current_stream().wait_stream(stream)
