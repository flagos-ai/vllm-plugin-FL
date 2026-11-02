# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("device_type", ["npu", "gcu"])
def test_device_weak_refs_are_recursive(monkeypatch, device_type):
    import torch

    from vllm.sequence import IntermediateTensors

    graph_module = import_module("vllm_fl.compilation.graph")

    weak_refs = {}

    def fake_weak_ref(tensor):
        weak_refs[id(tensor)] = object()
        return weak_refs[id(tensor)]

    monkeypatch.setattr(graph_module.current_platform, "device_type", device_type)
    monkeypatch.setattr(graph_module, f"_weak_ref_{device_type}_tensor", fake_weak_ref)

    first = torch.ones(1)
    second = torch.ones(2)
    third = torch.ones(3)
    empty = torch.ones(8)[:0]
    value = {
        "nested": [first, (second, empty)],
        "intermediate": IntermediateTensors({"hidden": third}),
        "metadata": None,
    }

    result = graph_module.weak_ref_tensors(value)

    assert result["nested"][0] is weak_refs[id(first)]
    assert result["nested"][1][0] is weak_refs[id(second)]
    assert result["nested"][1][1] is weak_refs[id(empty)]
    assert result["intermediate"].tensors["hidden"] is weak_refs[id(third)]
    assert result["metadata"] is None
    assert len(weak_refs) == 4


def test_npu_weak_ref_keeps_cpu_tensors(monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")

    def reject_cpu_tensor(_tensor):
        raise AssertionError("torch_npu weak refs must only receive NPU tensors")

    monkeypatch.setattr(graph_module.current_platform, "device_type", "npu")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(_C=SimpleNamespace(_weak_ref_tensor=reject_cpu_tensor)),
    )
    tensor = torch.ones(2)

    result = graph_module.weak_ref_tensors({"cpu": tensor})

    assert result["cpu"] is tensor


@pytest.mark.parametrize("primitive", [None, object()])
def test_npu_weak_ref_requires_runtime_primitive(monkeypatch, primitive):
    import sys

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = SimpleNamespace(type="npu")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(_C=SimpleNamespace(_weak_ref_tensor=primitive)),
    )

    with pytest.raises(RuntimeError, match="NPU graph weak references require"):
        graph_module._weak_ref_npu_tensor(tensor)


@pytest.mark.gpu
def test_npu_weak_ref_preserves_view_without_owning_storage():
    import gc
    import weakref

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    if graph_module.current_platform.device_type != "npu":
        pytest.skip("Requires an Ascend NPU")

    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    owner = torch.arange(32, device="npu", dtype=torch.float32).reshape(4, 8)
    view = owner[1:, 1::2]
    alias = graph_module._weak_ref_npu_tensor(view)
    empty_alias = graph_module._weak_ref_npu_tensor(view[:0])
    assert alias is not view
    assert alias.data_ptr() == view.data_ptr()
    assert alias.shape == view.shape
    assert alias.stride() == view.stride()
    assert alias.storage_offset() == view.storage_offset()
    assert alias.untyped_storage()._cdata != view.untyped_storage()._cdata
    assert empty_alias.shape == view[:0].shape
    assert empty_alias.stride() == view.stride()
    assert empty_alias.storage_offset() == view.storage_offset()
    torch.testing.assert_close(alias, view)

    owner_ref, view_ref = weakref.ref(owner), weakref.ref(view)
    del owner, view
    gc.collect()
    torch.npu.synchronize()
    assert owner_ref() is None
    assert view_ref() is None
    # The aliases remain alive but must not keep the original allocation alive.
    # Do not dereference their data after releasing the owner.
    assert torch.npu.memory_allocated() <= baseline


def test_npu_graph_capture_uses_active_accelerator_stream(monkeypatch):
    from types import SimpleNamespace

    graph_module = import_module("vllm_fl.compilation.graph")

    active_stream = object()
    stale_vllm_stream = MagicMock(return_value=object())
    npu_current_stream = MagicMock(return_value=active_stream)
    monkeypatch.setattr(graph_module.current_platform, "device_type", "npu")
    monkeypatch.setattr(
        graph_module.current_platform,
        "torch_device_fn",
        SimpleNamespace(current_stream=npu_current_stream),
    )
    monkeypatch.setattr(graph_module, "current_stream", stale_vllm_stream)

    assert graph_module._graph_capture_stream() is active_stream
    npu_current_stream.assert_called_once_with()
    stale_vllm_stream.assert_not_called()


def test_non_npu_graph_capture_preserves_vllm_stream(monkeypatch):
    graph_module = import_module("vllm_fl.compilation.graph")

    expected_stream = object()
    vllm_current_stream = MagicMock(return_value=expected_stream)
    monkeypatch.setattr(graph_module.current_platform, "device_type", "cuda")
    monkeypatch.setattr(graph_module, "current_stream", vllm_current_stream)

    assert graph_module._graph_capture_stream() is expected_stream
    vllm_current_stream.assert_called_once_with()


@pytest.mark.gpu
@pytest.mark.parametrize("weak_ref_output", [False, True])
def test_npu_graph_wrapper_capture_and_replay(monkeypatch, weak_ref_output):
    """Capture the real wrapper, then replay changed inputs into poisoned output."""
    import torch

    from vllm.config import CUDAGraphMode

    graph_module = import_module("vllm_fl.compilation.graph")
    if graph_module.current_platform.device_type != "npu":
        pytest.skip("Requires an Ascend NPU")
    import torch_npu  # noqa: F401 - registers torch.npu

    mode = CUDAGraphMode.FULL
    descriptor = object()
    forward_context = SimpleNamespace(
        batch_descriptor=descriptor, cudagraph_runtime_mode=mode
    )
    monkeypatch.setattr(
        graph_module, "validate_cudagraph_capturing_enabled", lambda: None
    )
    monkeypatch.setattr(graph_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(graph_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        graph_module, "validate_cudagraph_capturing_enabled", lambda: None
    )

    calls = []

    def eager_reference(x):
        return torch.relu(x * 2) + 3

    def runnable(x):
        calls.append(None)
        return eager_reference(x)

    wrapper = graph_module.GraphWrapper(
        runnable,
        SimpleNamespace(
            compilation_config=SimpleNamespace(), use_cumem_cudagraph_pool=False
        ),
        mode,
        graph_module.GraphOptions(weak_ref_output=weak_ref_output),
    )
    static_input = torch.arange(-4, 4, device="npu", dtype=torch.float32).reshape(2, 4)
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        captured_output = wrapper(static_input)
    torch.npu.current_stream().wait_stream(capture_stream)
    torch.npu.synchronize()

    entry = wrapper.concrete_graph_entries[descriptor]
    assert isinstance(entry.graph, torch.npu.NPUGraph)
    assert len(calls) == 1

    # NPU capture need not materialize its output until the first replay.
    captured_output.fill_(-999)
    torch.npu.synchronize()
    initial_replay = wrapper(static_input)
    torch.npu.synchronize()
    torch.testing.assert_close(initial_replay, eager_reference(static_input))
    assert len(calls) == 1

    for values in (
        torch.full_like(static_input, -2),
        torch.arange(8, device="npu", dtype=torch.float32).reshape(2, 4),
    ):
        static_input.copy_(values)
        captured_output.fill_(-999)
        torch.npu.synchronize()

        replayed_output = wrapper(static_input)
        torch.npu.synchronize()
        expected = eager_reference(values)
        torch.testing.assert_close(replayed_output, expected)
        assert not torch.all(replayed_output == -999).item()
        assert wrapper.concrete_graph_entries[descriptor].graph is entry.graph
        assert len(calls) == 1


class TestGraphOptions:
    """Test GraphOptions dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions()

        assert options.debug_log_enable is True
        assert options.gc_disable is False
        assert options.weak_ref_output is True

    def test_custom_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions(
            debug_log_enable=False,
            gc_disable=True,
            weak_ref_output=False,
        )

        assert options.debug_log_enable is False
        assert options.gc_disable is True
        assert options.weak_ref_output is False


class TestGraphEntry:
    """Test GraphEntry dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphEntry

        mock_batch_desc = MagicMock()

        entry = GraphEntry(batch_descriptor=mock_batch_desc)

        assert entry.batch_descriptor is mock_batch_desc
        assert entry.graph is None
        assert entry.output is None
        assert entry.input_addresses is None


def test_gcu_weak_ref_keeps_cpu_tensors(monkeypatch):
    import sys

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    monkeypatch.setattr(graph_module.current_platform, "device_type", "gcu")
    # CPU values must not require a vendor runtime primitive.
    monkeypatch.setitem(sys.modules, "torch_gcu", SimpleNamespace())
    value = torch.ones(2)
    assert graph_module.weak_ref_tensors({"cpu": value})["cpu"] is value


@pytest.mark.parametrize("missing", ["storage", "tensor"])
@pytest.mark.parametrize("primitive", [None, object()])
def test_gcu_weak_ref_requires_runtime_primitives(monkeypatch, missing, primitive):
    import sys

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = SimpleNamespace(type="gcu")
    storage_constructor = primitive if missing == "storage" else lambda *args: None
    tensor_constructor = primitive if missing == "tensor" else lambda *args: None
    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        storage_constructor,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        SimpleNamespace(
            _C=SimpleNamespace(
                _construct_GCU_Tensor_From_Storage_And_Metadata=tensor_constructor
            )
        ),
    )
    monkeypatch.setattr(graph_module.current_platform, "device_type", "gcu")
    with pytest.raises(RuntimeError, match="GCU graph weak references require"):
        graph_module.weak_ref_tensors(tensor)


def test_gcu_weak_ref_propagates_construction_failure(monkeypatch):
    import sys

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = SimpleNamespace(type="gcu")

    def incompatible(*args):
        raise ValueError("incompatible vendor tensor metadata")

    monkeypatch.setattr(
        torch._C,
        "_construct_storage_from_data_pointer",
        lambda *args: object(),
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        SimpleNamespace(
            _C=SimpleNamespace(
                _construct_GCU_Tensor_From_Storage_And_Metadata=incompatible
            )
        ),
    )
    monkeypatch.setattr(graph_module.current_platform, "device_type", "gcu")
    with pytest.raises(ValueError, match="incompatible vendor tensor metadata"):
        graph_module.weak_ref_tensors(tensor)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_gcu_weak_ref_preserves_views_without_owning_storage(dtype_name):
    import gc
    import weakref

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")
    if graph_module.current_platform.device_type != "gcu":
        pytest.skip("Requires an Enflame GCU")

    torch.gcu.synchronize()
    baseline = torch.gcu.memory_allocated()
    owner = torch.arange(32, device="gcu", dtype=getattr(torch, dtype_name)).reshape(
        4, 8
    )
    view = owner[1:, 1::2]
    alias = graph_module.weak_ref_tensors(view)
    empty = graph_module.weak_ref_tensors(view[:0])
    assert alias is not view
    assert alias.data_ptr() == view.data_ptr()
    assert alias.shape == view.shape and alias.stride() == view.stride()
    assert alias.storage_offset() == view.storage_offset()
    assert alias.untyped_storage()._cdata != view.untyped_storage()._cdata
    assert alias.untyped_storage().nbytes() == view.untyped_storage().nbytes()
    assert empty.shape == view[:0].shape and empty.stride() == view.stride()
    assert empty.storage_offset() == view.storage_offset()
    assert empty.untyped_storage()._cdata != view.untyped_storage()._cdata
    torch.testing.assert_close(alias.cpu(), view.cpu(), rtol=0, atol=0)

    owner_ref, view_ref = weakref.ref(owner), weakref.ref(view)
    del owner, view
    gc.collect()
    torch.gcu.synchronize()
    assert owner_ref() is None and view_ref() is None
    # Borrowed aliases stay alive but their original allocation must be released.
    # Never dereference either alias after its owning tensor/graph is released.
    assert torch.gcu.memory_allocated() <= baseline


def test_native_graph_registry_can_clear_fl_captures(monkeypatch):
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode

    graph_module = import_module("vllm_fl.compilation.graph")
    monkeypatch.setattr(
        graph_module.current_platform, "get_global_graph_pool", lambda: None
    )
    wrapper = graph_module.GraphWrapper(
        lambda: None,
        SimpleNamespace(
            compilation_config=SimpleNamespace(), use_cumem_cudagraph_pool=False
        ),
        CUDAGraphMode.FULL,
    )
    assert isinstance(wrapper, CUDAGraphWrapper)
    assert wrapper in CUDAGraphWrapper._all_instances
    assert wrapper in graph_module.GraphWrapper._all_instances
    wrapper.concrete_graph_entries["capture"] = object()
    CUDAGraphWrapper.clear_all_graphs()
    assert wrapper.concrete_graph_entries == {}


def test_native_graph_wrapper_is_replaced_without_changing_runner_type():
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_fl.compilation.graph import GraphWrapper, adapt_graph_wrapper

    model = object()
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(), use_cumem_cudagraph_pool=False
    )
    original = CUDAGraphWrapper(model, config, CUDAGraphMode.FULL)
    adapted = adapt_graph_wrapper(original)
    assert isinstance(adapted, GraphWrapper)
    assert isinstance(adapted, CUDAGraphWrapper)
    assert adapted in CUDAGraphWrapper._all_instances
    assert adapted in GraphWrapper._all_instances
    assert adapted.runtime_mode == CUDAGraphMode.FULL
    assert adapted.cudagraph_options is original.cudagraph_options
    runner = object.__new__(GPUModelRunner)
    runner.model = adapted
    assert runner.get_model() is model
    assert adapt_graph_wrapper(adapted) is adapted
    assert adapt_graph_wrapper(model) is model
    replacement = {}
    adapted.concrete_cudagraph_entries = replacement
    assert adapted.concrete_graph_entries is replacement


@pytest.mark.parametrize("device_type", ["cuda", "npu", "gcu"])
def test_all_platforms_share_the_graph_wrapper(monkeypatch, device_type):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", device_type)
    assert PlatformFL.get_static_graph_wrapper_cls() == (
        "vllm_fl.compilation.graph.GraphWrapper"
    )


@pytest.mark.parametrize("offloader_fails", [False, True])
def test_capture_uses_native_pool_context_and_preserves_offloader_errors(
    monkeypatch, offloader_fails
):
    from contextlib import contextmanager

    from vllm.config import CUDAGraphMode

    graph_module = import_module("vllm_fl.compilation.graph")
    events = []
    original_pool, capture_pool = (1, 2), (3, 4)

    @contextmanager
    def native_pool(pool, config):
        assert pool == original_pool
        events.append("pool-enter")
        try:
            yield capture_pool
        finally:
            events.append("pool-exit")

    @contextmanager
    def graph_context(graph, *, pool, stream):
        assert pool == capture_pool
        events.append("capture")
        yield

    def sync():
        if offloader_fails:
            raise RuntimeError("offloader failed")

    platform = SimpleNamespace(
        device_type="gcu",
        get_global_graph_pool=lambda: original_pool,
        torch_device_fn=SimpleNamespace(
            current_stream=lambda: None, graph=graph_context
        ),
    )
    descriptor = object()
    context = SimpleNamespace(
        batch_descriptor=descriptor, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    monkeypatch.setattr(graph_module, "current_platform", platform)
    monkeypatch.setattr(graph_module, "capture_pool", native_pool)
    monkeypatch.setattr(graph_module.Graph, "graph", lambda: object())
    monkeypatch.setattr(
        graph_module, "validate_cudagraph_capturing_enabled", lambda: None
    )
    monkeypatch.setattr(graph_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(graph_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(graph_module, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(
        graph_module,
        "get_offloader",
        lambda: SimpleNamespace(sync_prev_onload=sync, join_after_forward=lambda: None),
    )
    monkeypatch.setattr(graph_module.compilation_counter, "num_cudagraph_captured", 0)
    output = object()
    wrapper = graph_module.GraphWrapper(
        lambda: output,
        SimpleNamespace(
            compilation_config=SimpleNamespace(), use_cumem_cudagraph_pool=False
        ),
        CUDAGraphMode.FULL,
        graph_module.GraphOptions(debug_log_enable=False),
    )
    if offloader_fails:
        with pytest.raises(RuntimeError, match="offloader failed"):
            wrapper()
        assert events == ["pool-enter", "pool-exit"]
    else:
        assert wrapper() is output
        assert events == ["pool-enter", "capture", "pool-exit"]
        entry = wrapper.concrete_graph_entries[descriptor]
        assert entry.cudagraph is entry.graph
