import gc
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


def test_iluvatar_recursive_refs(monkeypatch):
    from vllm.sequence import IntermediateTensors

    module = import_module("vllm_fl.dispatch.backends.vendor.iluvatar.graph")
    monkeypatch.setattr(
        import_module("vllm_fl.compilation.graph"),
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="iluvatar"),
    )
    refs = {}

    def replace(tensor):
        return refs.setdefault(id(tensor), object())

    monkeypatch.setattr(module, "_weak_ref_iluvatar_tensor", replace)
    a, b = torch.ones(1), torch.ones(2)
    result = import_module("vllm_fl.compilation.graph").weak_ref_tensors(
        {"a": [a, (b,)], "pp": IntermediateTensors({"hidden": a}), "meta": None}
    )
    assert result["a"][0] is refs[id(a)]
    assert result["a"][1][0] is refs[id(b)]
    assert result["pp"].tensors["hidden"] is refs[id(a)]
    assert result["meta"] is None


def test_iluvatar_cpu_unchanged():
    module = import_module("vllm_fl.dispatch.backends.vendor.iluvatar.graph")
    tensor = torch.ones(2)
    assert module._weak_ref_iluvatar_tensor(tensor) is tensor


@pytest.mark.gpu
def test_iluvatar_borrowed_view_and_lifetime():
    module = import_module("vllm_fl.dispatch.backends.vendor.iluvatar.graph")
    if (
        getattr(
            import_module("vllm_fl.compilation.graph").current_platform,
            "vendor_name",
            None,
        )
        != "iluvatar"
    ):
        pytest.skip("Requires Iluvatar")
    x = torch.arange(64, device="cuda").reshape(8, 8)[1::2, ::2].T
    for value in (x, torch._neg_view(x)):
        ref = module._weak_ref_iluvatar_tensor(value)
        torch.testing.assert_close(ref, value)
        assert ref.storage_offset() == value.storage_offset()
        assert ref.stride() == value.stride()
        assert ref.is_neg() == value.is_neg()
    del x, value, ref
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    x = torch.ones((8192, 512), device="cuda", dtype=torch.bfloat16)
    ref = module._weak_ref_iluvatar_tensor(x)
    assert ref is not x
    del x
    gc.collect()
    assert torch.cuda.memory_allocated() == baseline
    del ref


@pytest.mark.gpu
def test_iluvatar_graph_ref_replay():
    module = import_module("vllm_fl.compilation.graph")
    if getattr(module.current_platform, "vendor_name", None) != "iluvatar":
        pytest.skip("Requires Iluvatar")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.ones(1024, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = x + 2
        ref = module.weak_ref_tensors(output)
        del output
        x.fill_(5)
        graph.replay()
        torch.testing.assert_close(ref, torch.full_like(ref, 7))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@pytest.mark.gpu
def test_iluvatar_graph_outputs_reuse_shared_pool():
    module = import_module("vllm_fl.compilation.graph")
    if getattr(module.current_platform, "vendor_name", None) != "iluvatar":
        pytest.skip("Requires Iluvatar")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.ones(4 * 1024 * 1024, device="cuda")
        pool = torch.cuda.graph_pool_handle()
        entries = []
        allocations = []
        for count in (4 * 1024 * 1024, 3 * 1024 * 1024, 2 * 1024 * 1024):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = x[:count] + 2
            ref = module.weak_ref_tensors(output)
            del output
            entries.append((graph, ref))
            allocations.append(torch.cuda.memory_allocated())
        # An owning output would retain another 12+8 MiB across these captures.
        assert allocations[-1] <= allocations[0] + 1024 * 1024
        x.fill_(9)
        for graph, ref in entries:
            graph.replay()
            torch.testing.assert_close(ref, torch.full_like(ref, 11))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def test_iluvatar_missing_primitives_fail_closed(monkeypatch):
    module = import_module("vllm_fl.dispatch.backends.vendor.iluvatar.graph")
    monkeypatch.setattr(torch._C, "_construct_storage_from_data_pointer", None)
    value = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    with pytest.raises(RuntimeError, match="Disable graph mode or update PyTorch"):
        module._weak_ref_iluvatar_tensor(value)


def test_nvidia_keeps_upstream_refs(monkeypatch):
    module = import_module("vllm_fl.compilation.graph")
    upstream = import_module("vllm.utils.torch_utils")
    monkeypatch.setattr(
        import_module("vllm_fl.compilation.graph"),
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="nvidia"),
    )
    sentinel = object()
    monkeypatch.setattr(upstream, "weak_ref_tensors", lambda value: sentinel)
    assert module.weak_ref_tensors(torch.ones(1)) is sentinel
