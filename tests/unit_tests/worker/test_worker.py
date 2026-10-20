# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the FL adaptation of vLLM v0.28.0 GPUWorker."""

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
    parallel_config = SimpleNamespace(
        worker_cls=None,
        all2all_backend=None,
        data_parallel_size=1,
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
