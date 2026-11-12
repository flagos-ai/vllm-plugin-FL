# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the FL adaptation of vLLM v0.28.0 GPUWorker."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def native_cuda_platform_imports_on_empty_wheels(request, monkeypatch):
    """Load the real CUDA platform for metadata contracts on empty wheels.

    These six tests patch CUDA operations themselves. The optional extension's
    import only registers native ops, which empty backend wheels do not ship.
    Keep import failures from an installed extension visible, and restore every
    module/package attribute introduced by this temporary import environment.
    """
    if not request.node.name.startswith("test_nvidia_platform_"):
        yield
        return

    import importlib.util
    import sys
    from types import ModuleType

    import torch

    import vllm
    import vllm.platforms

    import vllm_fl

    extension = "vllm._C_stable_libtorch"
    if extension in sys.modules or importlib.util.find_spec(extension) is not None:
        yield
        return

    imported_modules = ("vllm.platforms.cuda", "vllm_fl.nvidia_platform")
    previous_modules = {
        name: sys.modules[name] for name in imported_modules if name in sys.modules
    }
    missing = object()
    package_attributes = (
        (vllm, "_C_stable_libtorch"),
        (vllm.platforms, "cuda"),
        (vllm_fl, "nvidia_platform"),
    )
    previous_attributes = [
        (package, name, getattr(package, name, missing))
        for package, name in package_attributes
    ]
    cudnn_sdp = torch.backends.cuda.cudnn_sdp_enabled()
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, extension, ModuleType(extension))
        try:
            yield
        finally:
            for name in imported_modules:
                if name not in previous_modules:
                    sys.modules.pop(name, None)
            for package, name, value in previous_attributes:
                if value is missing:
                    if hasattr(package, name):
                        delattr(package, name)
                else:
                    setattr(package, name, value)
            torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)


def has_vllm_worker() -> bool:
    try:
        from vllm_fl.worker.worker import WorkerFL  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(
    not has_vllm_worker(), reason="vllm_fl.worker.worker is unavailable"
)


@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_ascend_does_not_request_cuda_memory_pool(monkeypatch, tag):
    from types import SimpleNamespace

    from vllm_fl.worker import worker as worker_module

    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="npu")
    )

    def unavailable_allocator():
        raise AssertionError("Ascend must not request the CUDA allocator")

    monkeypatch.setattr(
        worker_module, "get_mem_allocator_instance", unavailable_allocator
    )
    worker = worker_module.WorkerFL.__new__(worker_module.WorkerFL)
    with worker._maybe_get_memory_pool_context(tag):
        pass


def test_sunrise_process_group_uses_pccl(monkeypatch):
    from vllm_fl.worker import worker as worker_module

    monkeypatch.setattr(worker_module.current_platform, "device_type", "ptpu")

    assert worker_module._normalize_process_group_backend("flagcx") == "pccl"
    assert worker_module._normalize_process_group_backend("nccl") == "pccl"
    assert worker_module._normalize_process_group_backend("gloo") == "gloo"


def test_mapped_host_buffers_probe_the_cuda_view_op(monkeypatch):
    """The V2 UVA buffers need a real mapped-host-memory op, not just pinning."""
    import torch

    import vllm.utils.platform_utils as platform_utils
    import vllm.utils.torch_utils as torch_utils

    from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

    probed = []

    def unavailable_view(tensor):
        probed.append(tensor)
        raise RuntimeError("vLLM empty builds do not provide a mapped CUDA view")

    monkeypatch.setattr(
        torch_utils, "get_accelerator_view_from_cpu_tensor", unavailable_view
    )
    monkeypatch.setattr(
        torch, "zeros", lambda *args, **kwargs: torch.empty(1, dtype=torch.int32)
    )

    monkeypatch.setattr(platform_utils, "is_uva_available", lambda: False)
    assert v2_runner_compat.mapped_host_buffers_available() is False
    assert probed == []

    monkeypatch.setattr(platform_utils, "is_uva_available", lambda: True)
    assert v2_runner_compat.mapped_host_buffers_available() is False
    assert len(probed) == 1


@pytest.mark.parametrize(
    "device_type,flagcx_backend",
    [
        ("npu", "hccl"),
        ("cuda", "flagcx"),
        ("musa", "flagcx"),
        ("maca", "flagcx"),
        ("cpu", "flagcx"),
    ],
)
@pytest.mark.parametrize("backend", ["flagcx", "nccl", "pccl", "hccl", "gloo"])
def test_sunrise_process_group_mapping_is_ptpu_only(
    monkeypatch, device_type, flagcx_backend, backend
):
    from vllm_fl.worker import worker as worker_module

    monkeypatch.setattr(worker_module.current_platform, "device_type", device_type)

    # Preserve Ascend's own native process group when TP collectives use FlagCX.
    expected = flagcx_backend if backend == "flagcx" else backend
    assert worker_module._normalize_process_group_backend(backend) == expected


def test_worker_keeps_target_lifecycle_contract():
    from vllm_fl.worker.worker import WorkerFL

    required_methods = {
        "sleep",
        "wake_up",
        "checkpoint_prepare",
        "checkpoint_restore",
        "init_weight_transfer_engine",
        "start_weight_update",
        "start_draft_weight_update",
        "update_weights",
        "finish_weight_update",
        "elastic_ep_execute",
        "shutdown",
    }

    missing = sorted(name for name in required_methods if not hasattr(WorkerFL, name))
    assert not missing, f"WorkerFL is missing v0.28.0 methods: {missing}"


def test_worker_selects_v1_or_v2_model_runner():
    import inspect

    from vllm_fl.worker.worker import WorkerFL

    source = inspect.getsource(WorkerFL.init_device)
    assert "GPUModelRunnerV2" in source
    assert "vllm_fl.worker.model_runner" in source
    assert "ModelRunnerFL" in source


def test_platform_accepts_v2_model_runner(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    parallel_config = SimpleNamespace(
        worker_cls=None,
        all2all_backend=None,
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=None,
        scheduler_config=SimpleNamespace(),
        cache_config=None,
        compilation_config=SimpleNamespace(
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        attention_config=None,
        use_v2_model_runner=True,
    )

    PlatformFL.check_and_update_config(vllm_config)

    assert parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def test_v2_initializes_uva_before_upstream_buffers(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar import iluvatar
    from vllm_fl.dispatch.backends.vendor.iluvatar.worker import model_runner_v2

    events = []
    monkeypatch.setattr(iluvatar, "patch_v2_uva", lambda: events.append("uva"))
    monkeypatch.setattr(
        model_runner_v2.GPUModelRunner,
        "__init__",
        lambda self, config, device: events.append((config, device)),
    )
    model_runner_v2.IluvatarGPUModelRunnerV2("config", "device")
    assert events == ["uva", ("config", "device")]


def test_compat_does_not_hide_transitive_import_error(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.patches import worker_compat

    monkeypatch.setattr(worker_compat, "apply_worker_symbols_032", lambda: None)

    def broken(name):
        raise ModuleNotFoundError("missing dependency", name="unrelated_dependency")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        worker_compat.apply_import_compat()


def test_compat_preserves_existing_old_module(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.patches import worker_compat

    calls = []
    monkeypatch.setattr(worker_compat, "apply_worker_symbols_032", lambda: None)
    monkeypatch.setattr(importlib, "import_module", lambda name: calls.append(name))
    worker_compat.apply_import_compat()
    assert calls == ["vllm.model_executor.layers.batch_invariant"]


def test_vendor_update_config_synchronizes_worker_and_runner(monkeypatch):
    import vllm.config

    from vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker import IluvatarWorkerFL

    new_config = object()
    monkeypatch.setattr(vllm.config, "update_config", lambda old, values: new_config)
    worker = object.__new__(IluvatarWorkerFL)
    worker.vllm_config = SimpleNamespace(model_config=object())
    worker.model_runner = SimpleNamespace()
    worker.update_config({"model_config": {"dtype": "auto"}})
    assert worker.model_config is new_config
    assert worker.model_runner.model_config is new_config
    assert worker.vllm_config.model_config is new_config
    with pytest.raises(ValueError, match="not supported"):
        worker.update_config({"unsupported": {}})


def test_compat_aliases_missing_module_and_is_idempotent(monkeypatch):
    import sys

    from vllm_fl.dispatch.backends.vendor.iluvatar.patches import worker_compat

    old = "vllm.model_executor.layers.batch_invariant"
    new = "vllm.model_executor.determinism.batch_invariant"
    replacement = SimpleNamespace(init_batch_invariance=object())
    calls = []
    monkeypatch.delitem(sys.modules, old, raising=False)
    # Ensure pytest restores the real module after this test mutates the alias.
    monkeypatch.setitem(sys.modules, old, None)
    monkeypatch.setattr(worker_compat, "apply_worker_symbols_032", lambda: None)

    def load(name):
        calls.append(name)
        if name == old:
            module = sys.modules.get(old)
            if module is None:
                raise ModuleNotFoundError(name, name=old)
            return module
        assert name == new
        return replacement

    monkeypatch.setattr(importlib, "import_module", load)
    worker_compat.apply_import_compat()
    worker_compat.apply_import_compat()
    assert sys.modules[old] is replacement
    assert calls == [old, new, old]


def test_worker_installs_pipeline_after_base_initialization(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar import iluvatar, pipeline
    from vllm_fl.dispatch.backends.vendor.iluvatar.worker import worker

    events = []
    group = object()
    monkeypatch.setattr(
        iluvatar, "apply_worker_patches", lambda: events.append("patch")
    )
    monkeypatch.setattr(
        worker.BaseWorkerFL, "init_device", lambda self: events.append("base")
    )
    monkeypatch.setattr(worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        pipeline, "install_pipeline_transport", lambda value: events.append(value)
    )
    object.__new__(worker.IluvatarWorkerFL).init_device()
    assert events == ["patch", "base", group]


@pytest.mark.parametrize("vendor", [None, "nvidia", "ascend", "iluvatar"])
def test_shared_v2_route_preserves_upstream_for_other_platforms(vendor):
    import ast
    import inspect

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2 import (
        IluvatarGPUModelRunnerV2,
    )
    from vllm_fl.worker import worker

    # Execute the production routing block without allocating GPU worker state.
    tree = ast.parse(inspect.getsource(worker))
    routes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module
            == "vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2"
            for child in node.body
        )
    ]
    assert len(routes) == 1
    platform = (
        SimpleNamespace() if vendor is None else SimpleNamespace(vendor_name=vendor)
    )
    namespace = {"current_platform": platform, "GPUModelRunnerV2": GPUModelRunner}
    exec(  # noqa: S102 - execute only the production runner selection block
        compile(ast.Module(body=routes, type_ignores=[]), worker.__file__, "exec"),
        namespace,
    )
    expected = IluvatarGPUModelRunnerV2 if vendor == "iluvatar" else GPUModelRunner
    assert namespace["GPUModelRunnerV2"] is expected
