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


@pytest.fixture
def memory_pool_worker(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    import vllm_fl.worker.worker as worker_module

    def make_worker(platform, *, enable_cumem_allocator, enable_sleep_mode):
        monkeypatch.setattr(
            worker_module,
            "current_platform",
            SimpleNamespace(
                vendor_name=platform,
                is_cuda_alike=lambda: platform == "cuda",
                is_xpu=lambda: platform == "xpu",
                is_cpu=lambda: platform == "cpu",
            ),
        )
        worker = object.__new__(worker_module.WorkerFL)
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                enable_cumem_allocator=enable_cumem_allocator,
                enable_sleep_mode=enable_sleep_mode,
            )
        )
        allocator = Mock()
        allocator.get_current_usage.return_value = 0
        allocator.use_memory_pool.return_value = nullcontext("allocator")
        get_allocator = Mock(return_value=allocator)
        monkeypatch.setattr(worker_module, "get_mem_allocator_instance", get_allocator)
        return worker, get_allocator, allocator

    return make_worker


@pytest.mark.parametrize(
    ("platform", "enable_cumem_allocator", "enable_sleep_mode"),
    [
        ("hygon", False, False),
        ("hygon", True, False),
        ("cuda", False, False),
        ("cuda", False, True),
        ("xpu", False, False),
        ("xpu", True, False),
        ("cpu", False, False),
        ("cpu", True, True),
    ],
)
def test_worker_uses_ordinary_memory_without_allocator(
    memory_pool_worker, platform, enable_cumem_allocator, enable_sleep_mode
):
    worker, get_allocator, allocator = memory_pool_worker(
        platform,
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=enable_sleep_mode,
    )

    with worker._maybe_get_memory_pool_context("weights") as context:
        assert context is None

    get_allocator.assert_not_called()
    allocator.use_memory_pool.assert_not_called()


@pytest.mark.parametrize(
    ("platform", "enable_cumem_allocator", "enable_sleep_mode"),
    [
        ("cuda", True, False),
        ("cuda", True, True),
        ("xpu", False, True),
        ("xpu", True, True),
    ],
)
@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_worker_keeps_supported_allocator_selection(
    memory_pool_worker, platform, enable_cumem_allocator, enable_sleep_mode, tag
):
    worker, get_allocator, allocator = memory_pool_worker(
        platform,
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=enable_sleep_mode,
    )

    with worker._maybe_get_memory_pool_context(tag) as context:
        assert context == "allocator"

    get_allocator.assert_called_once_with()
    allocator.use_memory_pool.assert_called_once_with(tag=tag)
    if tag == "weights":
        allocator.get_current_usage.assert_called_once_with()
    else:
        allocator.get_current_usage.assert_not_called()


@pytest.mark.parametrize("enable_cumem_allocator", [False, True])
def test_hygon_explicit_sleep_preserves_unsupported_allocator_error(
    memory_pool_worker, enable_cumem_allocator
):
    worker, get_allocator, allocator = memory_pool_worker(
        "hygon",
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=True,
    )
    get_allocator.side_effect = RuntimeError("Sleep mode allocator is not available")

    with pytest.raises(RuntimeError, match="Sleep mode allocator is not available"):
        worker._maybe_get_memory_pool_context("weights")

    get_allocator.assert_called_once_with()
    allocator.use_memory_pool.assert_not_called()


def test_platform_accepts_v2_model_runner():
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode

    from vllm_fl.platform import PlatformFL

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
    from vllm.platforms import PlatformEnum
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert issubclass(NvidiaPlatformFL, CudaPlatform)
    assert NvidiaPlatformFL._enum == PlatformEnum.CUDA
    platform = NvidiaPlatformFL()
    assert platform.is_cuda()
    assert not platform.is_out_of_tree()


def test_nvidia_platform_selects_target_version_worker_wrapper(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    parallel_config = SimpleNamespace(
        worker_cls=None,
        disable_custom_all_reduce=True,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    with monkeypatch.context() as state:
        state.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
        state.delenv("FLAGCX_PATH", raising=False)
        with patch.object(CudaPlatform, "check_and_update_config") as native_update:
            NvidiaPlatformFL.check_and_update_config(vllm_config)

        assert NvidiaPlatformFL.dist_backend == CudaPlatform.dist_backend

    assert parallel_config.worker_cls == "vllm_fl.worker.worker.NvidiaWorkerFL"
    assert parallel_config.disable_custom_all_reduce is True
    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_keeps_native_cuda_communication(monkeypatch):
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    with monkeypatch.context() as state:
        state.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
        state.delenv("FLAGCX_PATH", raising=False)
        with (
            patch.object(
                CudaPlatform,
                "get_device_communicator_cls",
                return_value="native.cuda.Communicator",
            ) as native_communicator,
            patch.object(
                CudaPlatform,
                "use_custom_allreduce",
                return_value=True,
            ) as native_custom_allreduce,
        ):
            communicator = NvidiaPlatformFL.get_device_communicator_cls()
            use_custom_allreduce = NvidiaPlatformFL.use_custom_allreduce()

        assert NvidiaPlatformFL.dist_backend == CudaPlatform.dist_backend

    assert communicator == "native.cuda.Communicator"
    assert use_custom_allreduce is True
    native_communicator.assert_called_once_with()
    native_custom_allreduce.assert_called_once_with()


def test_nvidia_platform_uses_flagcx_when_configured(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    parallel_config = SimpleNamespace(
        worker_cls=None,
        disable_custom_all_reduce=False,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    # Importing the platform before changing the environment mirrors unit tests
    # that dynamically select FlagCX in an already-running Python process.
    with monkeypatch.context() as state:
        state.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
        state.setenv("FLAGCX_PATH", "/opt/flagcx")
        with patch.object(CudaPlatform, "check_and_update_config") as native_update:
            NvidiaPlatformFL.check_and_update_config(vllm_config)

        assert NvidiaPlatformFL.dist_backend == "flagcx"
        assert parallel_config.disable_custom_all_reduce is True
        assert (
            NvidiaPlatformFL.get_device_communicator_cls()
            == "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
        )
        assert NvidiaPlatformFL.use_custom_allreduce() is False

    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_uses_native_attention_by_default(monkeypatch):
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
