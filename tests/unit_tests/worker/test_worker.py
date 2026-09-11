# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the FL adaptation of vLLM v0.28.0 GPUWorker."""

import pytest


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

    import vllm_fl.worker.worker as worker_module

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


def test_nvidia_platform_keeps_native_cuda_semantics():
    pytest.importorskip("vllm._C_stable_libtorch", exc_type=ImportError)
    from vllm.platforms import PlatformEnum
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert issubclass(NvidiaPlatformFL, CudaPlatform)
    assert NvidiaPlatformFL._enum == PlatformEnum.CUDA
    platform = NvidiaPlatformFL()
    assert platform.is_cuda()
    assert not platform.is_out_of_tree()


def test_nvidia_platform_selects_target_version_worker_wrapper():
    pytest.importorskip("vllm._C_stable_libtorch", exc_type=ImportError)
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    parallel_config = SimpleNamespace(worker_cls=None)
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    with patch.object(CudaPlatform, "check_and_update_config") as native_update:
        NvidiaPlatformFL.check_and_update_config(vllm_config)

    assert parallel_config.worker_cls == "vllm_fl.worker.worker.NvidiaWorkerFL"
    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_uses_native_attention_by_default(monkeypatch):
    pytest.importorskip("vllm._C_stable_libtorch", exc_type=ImportError)
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    selector = SimpleNamespace(use_mla=False, use_sparse=False)

    with patch.object(
        CudaPlatform,
        "get_attn_backend_cls",
        return_value="native.attention.Backend",
    ) as native_select:
        result = NvidiaPlatformFL.get_attn_backend_cls(None, selector, 32)

    assert result == "native.attention.Backend"
    native_select.assert_called_once_with(None, selector, 32)


def test_nvidia_platform_honors_explicit_flaggems_attention(monkeypatch):
    pytest.importorskip("vllm._C_stable_libtorch", exc_type=ImportError)
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    selector = SimpleNamespace(use_mla=False, use_sparse=False)
    backend_path = (
        "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
    )

    with patch.object(
        FlagGemsBackend,
        "attention_backend",
        return_value=backend_path,
    ) as flaggems_select:
        result = NvidiaPlatformFL.get_attn_backend_cls(None, selector, 32)

    assert result == backend_path
    flaggems_select.assert_called_once_with(use_mla=False, use_sparse=False)


def test_nvidia_worker_delegates_to_target_version_gpu_worker():
    from vllm.v1.worker.gpu_worker import Worker as NativeGPUWorker

    from vllm_fl.worker.worker import NvidiaWorkerFL

    assert issubclass(NvidiaWorkerFL, NativeGPUWorker)


def test_native_runner_io_bridge_replaces_only_inference_mode():
    import torch

    from vllm_fl.dispatch.io_common import set_io_active
    from vllm_fl.worker.worker import _install_native_runner_io_methods

    class NativeRunner:
        @torch.inference_mode()
        def execute_model(self):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        @torch.inference_mode()
        def sample_tokens(self):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    runner = NativeRunner()
    _install_native_runner_io_methods(runner)

    try:
        set_io_active(True)
        assert runner.execute_model() == (False, False)
        assert runner.sample_tokens() == (False, False)

        set_io_active(False)
        assert runner.execute_model() == (True, False)
    finally:
        set_io_active(False)


def test_nvidia_worker_initializes_io_dump_once_after_model_load():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from vllm_fl.worker.worker import NvidiaWorkerFL

    worker = object.__new__(NvidiaWorkerFL)
    worker._fl_io_dump_initialized = False
    worker.model_config = SimpleNamespace(enforce_eager=True)
    model = object()
    worker.model_runner = SimpleNamespace(get_model=Mock(return_value=model))

    with (
        patch("vllm_fl.dispatch.io_dumper.init_io_dump_from_env") as init_io_dump,
        patch("vllm_fl.dispatch.io_dumper.is_dump_enabled", return_value=True),
        patch("vllm_fl.dispatch.io_dumper.register_io_module_hooks") as register_hooks,
        patch(
            "vllm_fl.worker.worker._install_native_runner_io_methods"
        ) as install_methods,
    ):
        worker._ensure_io_dump_initialized()
        worker._ensure_io_dump_initialized()

    init_io_dump.assert_called_once_with(True)
    install_methods.assert_called_once_with(worker.model_runner)
    register_hooks.assert_called_once_with(model)
