# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
    KunlunxinMetadata,
    _prepare_full_graph_kv_write,
)
from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
    patch_breakable_cudagraph_mode,
    patch_breakable_private_pools,
    patch_cudagraph_dispatcher,
    patch_decode_attention,
    patch_eager_all_gather,
    patch_graph_all_reduce,
)

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "kunlunxin",
    reason="Kunlunxin-specific CUDA graph tests",
)


def make_dispatcher(tensor_parallel_size=1):
    dispatcher = object.__new__(CudagraphDispatcher)
    dispatcher.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tensor_parallel_size)
    )
    return dispatcher


def test_exact_decode_keeps_full_available(monkeypatch):
    calls = []

    def dispatch(self, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(CudagraphDispatcher, "dispatch", dispatch)
    patch_cudagraph_dispatcher()

    dispatcher = make_dispatcher()
    dispatcher._bs_to_padded_graph_size = [0, 1, 2, 4, 4]
    result = dispatcher.dispatch(num_tokens=4, uniform_decode=True)

    assert result["invalid_modes"] is None
    assert calls[-1]["num_tokens"] == 4


def test_padded_decode_keeps_full_available(monkeypatch):
    def dispatch(self, **kwargs):
        return kwargs

    monkeypatch.setattr(CudagraphDispatcher, "dispatch", dispatch)
    patch_cudagraph_dispatcher()

    dispatcher = make_dispatcher()
    dispatcher._bs_to_padded_graph_size = [0, 1, 2, 4, 4]
    result = dispatcher.dispatch(num_tokens=3, uniform_decode=True)

    assert result["invalid_modes"] is None
    assert result["valid_modes"] is None


def test_non_uniform_batch_excludes_full(monkeypatch):
    def dispatch(self, **kwargs):
        return kwargs

    monkeypatch.setattr(CudagraphDispatcher, "dispatch", dispatch)
    patch_cudagraph_dispatcher()

    dispatcher = make_dispatcher()
    dispatcher._bs_to_padded_graph_size = [0, 1, 2]
    result = dispatcher.dispatch(num_tokens=2, uniform_decode=False)

    assert result["invalid_modes"] == {CUDAGraphMode.FULL}


def test_padded_decode_accepts_full_only_redispatch(monkeypatch):
    def dispatch(self, **kwargs):
        return kwargs

    monkeypatch.setattr(CudagraphDispatcher, "dispatch", dispatch)
    patch_cudagraph_dispatcher()

    dispatcher = make_dispatcher()
    dispatcher._bs_to_padded_graph_size = [0, 1, 2, 4, 4]
    result = dispatcher.dispatch(
        num_tokens=3,
        uniform_decode=True,
        valid_modes={CUDAGraphMode.FULL},
    )

    assert result["valid_modes"] == {CUDAGraphMode.FULL}
    assert result["invalid_modes"] is None


def test_tensor_parallel_exact_decode_keeps_full(monkeypatch):
    def dispatch(self, **kwargs):
        return kwargs

    monkeypatch.setattr(CudagraphDispatcher, "dispatch", dispatch)
    patch_cudagraph_dispatcher()

    dispatcher = make_dispatcher(tensor_parallel_size=4)
    dispatcher._bs_to_padded_graph_size = [0, 1, 2, 4, 4]
    result = dispatcher.dispatch(num_tokens=4, uniform_decode=True)

    assert result["invalid_modes"] is None


def test_full_graph_padding_writes_only_zeroes_to_null_block():
    key = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    value = key + 10
    slots = torch.tensor([8, 9, -1, -1], dtype=torch.int64)

    safe_key, safe_value, safe_slots, valid = _prepare_full_graph_kv_write(
        key, value, slots, block_size=4
    )

    assert torch.equal(safe_slots, torch.tensor([8, 9, 2, 3]))
    assert torch.equal(valid, torch.tensor([True, True, False, False]))
    assert torch.equal(safe_key[:2], key[:2])
    assert torch.equal(safe_value[:2], value[:2])
    assert torch.count_nonzero(safe_key[2:]) == 0
    assert torch.count_nonzero(safe_value[2:]) == 0


def test_full_graph_uses_paged_decode_and_eager_uses_prefix(monkeypatch):
    import xtorch_ops

    import vllm_fl.dispatch.backends.vendor.kunlunxin.patch as patch_mod
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinPagedAttention,
    )

    calls = []

    def paged_decode(*args, **kwargs):
        calls.append("paged")
        args[6].fill_(1)

    def prefix_decode(*args, **kwargs):
        calls.append("prefix")
        args[3].fill_(2)

    monkeypatch.setattr(xtorch_ops, "decode_paged_attention", paged_decode)
    monkeypatch.setattr(xtorch_ops, "prefill_attention", prefix_decode)
    patch_decode_attention()

    query = torch.zeros((2, 1, 4))
    cache = torch.zeros((1, 1, 2, 4))
    block_tables = torch.zeros((2, 1), dtype=torch.int32)
    seq_lens = torch.tensor([3, 4], dtype=torch.int32)
    output = torch.empty_like(query)
    lod = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_lod = torch.tensor([0, 3, 7], dtype=torch.int32)

    monkeypatch.setattr(patch_mod, "_is_full_graph_runtime", lambda: True)
    KunlunxinPagedAttention.forward_decode(
        query,
        cache,
        cache,
        block_tables,
        seq_lens,
        seq_lens,
        4,
        2,
        "auto",
        1,
        0.5,
        None,
        torch.tensor(1.0),
        torch.tensor(1.0),
        output=output,
        query_start_loc=lod,
        query_start_loc_host=lod,
        kv_prefix_start_loc=kv_lod,
        kv_prefix_start_loc_host=kv_lod,
    )
    assert calls == ["paged"]
    assert torch.count_nonzero(output - 1) == 0

    monkeypatch.setattr(patch_mod, "_is_full_graph_runtime", lambda: False)
    KunlunxinPagedAttention.forward_decode(
        query,
        cache,
        cache,
        block_tables,
        seq_lens,
        seq_lens,
        4,
        2,
        "auto",
        1,
        0.5,
        None,
        torch.tensor(1.0),
        torch.tensor(1.0),
        output=output,
        query_start_loc=lod,
        query_start_loc_host=lod,
        kv_prefix_start_loc=kv_lod,
        kv_prefix_start_loc_host=kv_lod,
    )
    assert calls == ["paged", "prefix"]
    assert torch.count_nonzero(output - 2) == 0


def test_mixed_batch_decode_metadata_excludes_prefill_kv_lod():
    metadata = KunlunxinMetadata(
        seq_lens_tensor=torch.tensor([25, 24, 26], dtype=torch.int32),
        max_decode_seq_len=1,
        block_tables=torch.zeros((3, 4), dtype=torch.int32),
        num_prefills=1,
        num_prefill_tokens=2,
        num_decode_tokens=2,
        slot_mapping=torch.arange(4, dtype=torch.int64),
        enable_kv_scales_calculation=False,
        max_prefill_seq_len=2,
        num_actual_tokens=4,
        use_cuda_graph=False,
        seq_lens_tensor_host=torch.tensor([25, 24, 26], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2, 4], dtype=torch.int32),
        query_start_loc_host=torch.tensor([0, 1, 2, 4], dtype=torch.int32),
        kv_prefix_start_loc=torch.tensor([0, 25, 49, 75], dtype=torch.int32),
        kv_prefix_start_loc_host=torch.tensor([0, 25, 49, 75], dtype=torch.int32),
    )

    decode_metadata = metadata.decode_metadata

    assert torch.equal(
        decode_metadata.query_start_loc_host,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        decode_metadata.kv_prefix_start_loc_host,
        torch.tensor([0, 25, 49], dtype=torch.int32),
    )
    assert torch.equal(
        decode_metadata.kv_prefix_start_loc,
        torch.tensor([0, 25, 49], dtype=torch.int32),
    )


def test_breakable_mode_is_enabled_without_environment_switch(monkeypatch):
    import vllm.compilation.breakable_cudagraph as breakable
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    import vllm_fl.worker.model_runner as fl_model_runner

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    disabled = lambda: False
    monkeypatch.setattr(breakable, "is_breakable_cudagraph_enabled", disabled)
    monkeypatch.setattr(gpu_model_runner, "is_breakable_cudagraph_enabled", disabled)
    monkeypatch.setattr(fl_model_runner, "is_breakable_cudagraph_enabled", disabled)

    patch_breakable_cudagraph_mode()

    assert breakable.is_breakable_cudagraph_enabled()
    assert gpu_model_runner.is_breakable_cudagraph_enabled()
    assert fl_model_runner.is_breakable_cudagraph_enabled()
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in __import__("os").environ


def test_breakable_full_graphs_do_not_share_memory_pool(monkeypatch):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    def init_with_shared_pool(self):
        self.graph_pool = "shared-pool"

    monkeypatch.setattr(BreakableCUDAGraphWrapper, "__init__", init_with_shared_pool)
    patch_breakable_private_pools()

    wrapper = object.__new__(BreakableCUDAGraphWrapper)
    BreakableCUDAGraphWrapper.__init__(wrapper)

    assert wrapper.graph_pool is None


def test_full_graph_all_reduce_uses_process_group(monkeypatch):
    import torch.distributed as dist

    import vllm.compilation.breakable_cudagraph as breakable
    import vllm.forward_context as forward_context

    from vllm_fl.distributed.communicator import CommunicatorFL

    eager_calls = []
    process_group_calls = []

    def eager_all_reduce(self, input_):
        eager_calls.append(input_)
        return input_ + 10

    def process_group_all_reduce(output, group):
        process_group_calls.append((output, group))
        output.add_(2)

    class Capture:
        _capturing = True

        def __init__(self):
            self.end_calls = 0
            self.begin_calls = 0

        def _end_segment(self):
            self.end_calls += 1
            self._capturing = False

        def _begin_segment(self):
            self.begin_calls += 1
            self._capturing = True

    monkeypatch.setattr(CommunicatorFL, "all_reduce", eager_all_reduce)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    monkeypatch.setattr(dist, "all_reduce", process_group_all_reduce)
    capture = Capture()
    monkeypatch.setattr(
        breakable.BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: capture),
    )
    patch_graph_all_reduce()

    communicator = SimpleNamespace(
        device_group="device-group", _kunlunxin_graph_pg_warmed=True
    )
    input_ = torch.tensor([1.0, 3.0])
    output = CommunicatorFL.all_reduce(communicator, input_)

    assert eager_calls == []
    assert len(process_group_calls) == 1
    assert process_group_calls[0][1] == "device-group"
    assert capture.end_calls == 0
    assert capture.begin_calls == 0
    assert output.data_ptr() != input_.data_ptr()
    assert torch.equal(output, torch.tensor([3.0, 5.0]))


def test_full_graph_all_reduce_warms_process_group_outside_capture(monkeypatch):
    import torch.distributed as dist

    import vllm.compilation.breakable_cudagraph as breakable
    import vllm.forward_context as forward_context

    from vllm_fl.distributed.communicator import CommunicatorFL

    process_group_calls = []
    synchronize_calls = []

    def process_group_all_reduce(output, group):
        process_group_calls.append((output.clone(), group))
        output.add_(2)

    class Capture:
        _capturing = True

        def __init__(self):
            self.end_calls = 0
            self.begin_calls = 0

        def _end_segment(self):
            self.end_calls += 1
            self._capturing = False

        def _begin_segment(self):
            self.begin_calls += 1
            self._capturing = True

    monkeypatch.setattr(CommunicatorFL, "all_reduce", lambda self, input_: input_ + 10)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: synchronize_calls.append(True)
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    monkeypatch.setattr(dist, "all_reduce", process_group_all_reduce)
    capture = Capture()
    monkeypatch.setattr(
        breakable.BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: capture),
    )
    patch_graph_all_reduce()

    communicator = SimpleNamespace(device_group="device-group")
    input_ = torch.tensor([1.0, 3.0])
    output = CommunicatorFL.all_reduce(communicator, input_)

    assert len(process_group_calls) == 2
    assert all(call[1] == "device-group" for call in process_group_calls)
    assert synchronize_calls == [True]
    assert capture.end_calls == 1
    assert capture.begin_calls == 1
    assert communicator._kunlunxin_graph_pg_warmed
    assert torch.equal(output, torch.tensor([3.0, 5.0]))


def test_full_graph_all_reduce_splits_after_sixteen_collectives(monkeypatch):
    import torch.distributed as dist

    import vllm.compilation.breakable_cudagraph as breakable
    import vllm.forward_context as forward_context

    from vllm_fl.distributed.communicator import CommunicatorFL

    class Capture:
        _capturing = True

        def __init__(self):
            self.end_calls = 0
            self.begin_calls = 0

        def _end_segment(self):
            self.end_calls += 1
            self._capturing = False

        def _begin_segment(self):
            self.begin_calls += 1
            self._capturing = True

    monkeypatch.setattr(CommunicatorFL, "all_reduce", lambda self, input_: input_ + 10)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    monkeypatch.setattr(dist, "all_reduce", lambda output, group: None)
    capture = Capture()
    monkeypatch.setattr(
        breakable.BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: capture),
    )
    patch_graph_all_reduce()

    communicator = SimpleNamespace(
        device_group="device-group", _kunlunxin_graph_pg_warmed=True
    )
    for _ in range(16):
        CommunicatorFL.all_reduce(communicator, torch.tensor([1.0]))

    assert capture.end_calls == 1
    assert capture.begin_calls == 1
    assert capture._kunlunxin_segment_all_reduces == 0


def test_piecewise_capture_keeps_direct_flagcx_all_reduce(monkeypatch):
    import torch.distributed as dist

    import vllm.forward_context as forward_context

    from vllm_fl.distributed.communicator import CommunicatorFL

    eager_calls = []
    process_group_calls = []

    def eager_all_reduce(self, input_):
        eager_calls.append(input_)
        return input_ + 10

    def process_group_all_reduce(output, group):
        process_group_calls.append((output, group))

    monkeypatch.setattr(CommunicatorFL, "all_reduce", eager_all_reduce)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    )
    monkeypatch.setattr(dist, "all_reduce", process_group_all_reduce)
    patch_graph_all_reduce()

    input_ = torch.tensor([1.0, 3.0])
    output = CommunicatorFL.all_reduce(SimpleNamespace(), input_)

    assert len(eager_calls) == 1
    assert torch.equal(eager_calls[0], input_)
    assert process_group_calls == []
    assert torch.equal(output, torch.tensor([11.0, 13.0]))


def test_eager_all_reduce_keeps_direct_flagcx_path(monkeypatch):
    from vllm_fl.distributed.communicator import CommunicatorFL

    eager_calls = []

    def eager_all_reduce(self, input_):
        eager_calls.append(input_)
        return input_ + 10

    monkeypatch.setattr(CommunicatorFL, "all_reduce", eager_all_reduce)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    patch_graph_all_reduce()

    input_ = torch.tensor([1.0, 3.0])
    output = CommunicatorFL.all_reduce(SimpleNamespace(), input_)

    assert len(eager_calls) == 1
    assert torch.equal(eager_calls[0], input_)
    assert torch.equal(output, torch.tensor([11.0, 13.0]))


def test_eager_all_gather_uses_direct_flagcx_without_process_group_events(
    monkeypatch,
):
    from vllm_fl.distributed.communicator import CommunicatorFL

    process_group_calls = []

    def process_group_all_gather(self, input_, dim=-1):
        process_group_calls.append((input_, dim))
        return input_ + 100

    class DirectFlagcx:
        disabled = False

        def __init__(self):
            self.calls = []

        def all_gather(self, output, input_):
            self.calls.append((output, input_))
            output.copy_(torch.cat((input_, input_ + 10), dim=0))

    monkeypatch.setattr(CommunicatorFL, "all_gather", process_group_all_gather)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    patch_eager_all_gather()

    direct = DirectFlagcx()
    communicator = SimpleNamespace(world_size=2, pyflagcx_comm=direct)
    input_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = CommunicatorFL.all_gather(communicator, input_, dim=1)

    assert process_group_calls == []
    assert len(direct.calls) == 1
    assert torch.equal(
        output,
        torch.tensor([[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]]),
    )


def test_capture_all_gather_keeps_process_group_path(monkeypatch):
    from vllm_fl.distributed.communicator import CommunicatorFL

    process_group_calls = []

    def process_group_all_gather(self, input_, dim=-1):
        process_group_calls.append((input_, dim))
        return input_ + 100

    monkeypatch.setattr(CommunicatorFL, "all_gather", process_group_all_gather)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    patch_eager_all_gather()

    input_ = torch.tensor([1.0, 3.0])
    communicator = SimpleNamespace(
        world_size=2,
        pyflagcx_comm=SimpleNamespace(disabled=False),
    )
    output = CommunicatorFL.all_gather(communicator, input_, dim=0)

    assert len(process_group_calls) == 1
    gathered_input, gathered_dim = process_group_calls[0]
    assert torch.equal(gathered_input, input_)
    assert gathered_dim == 0
    assert torch.equal(output, torch.tensor([101.0, 103.0]))
