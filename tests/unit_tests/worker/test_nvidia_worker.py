# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA worker/platform contracts for the native vLLM CUDA wheel."""

import pytest

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA-specific test"
)


@pytest.fixture(scope="module", autouse=True)
def initialize_nvidia_platform():
    # Resolve the real plugin entry point before importing CUDA internals.
    from vllm.platforms import current_platform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert isinstance(current_platform, NvidiaPlatformFL)


def test_nvidia_platform_keeps_native_cuda_semantics():
    from vllm.platforms import PlatformEnum
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert issubclass(NvidiaPlatformFL, CudaPlatform)
    assert NvidiaPlatformFL._enum == PlatformEnum.CUDA
    platform = NvidiaPlatformFL()
    assert platform.is_cuda()
    assert not platform.is_out_of_tree()


@pytest.mark.parametrize("use_v2", [False, True])
def test_nvidia_platform_selects_target_version_worker_wrapper(monkeypatch, use_v2):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    parallel_config = SimpleNamespace(
        worker_cls=None,
        disable_custom_all_reduce=True,
    )
    vllm_config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        parallel_config=parallel_config,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=False)
        ),
    )

    with monkeypatch.context() as state:
        state.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
        state.delenv("FLAGCX_PATH", raising=False)
        with patch.object(CudaPlatform, "check_and_update_config") as native_update:
            NvidiaPlatformFL.check_and_update_config(vllm_config)

        assert NvidiaPlatformFL.dist_backend == CudaPlatform.dist_backend

    assert vllm_config.use_v2_model_runner is use_v2
    assert parallel_config.worker_cls == "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"
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
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=False)
        ),
    )

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

    from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL

    assert issubclass(NvidiaWorkerFL, NativeGPUWorker)
    assert NvidiaWorkerFL.load_model is NativeGPUWorker.load_model


def test_native_runner_io_bridge_replaces_only_inference_mode():
    import torch

    from vllm_fl.dispatch.io_common import set_io_active
    from vllm_fl.worker.nvidia_worker import _install_native_runner_io_methods

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

    from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL

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
            "vllm_fl.worker.nvidia_worker._install_native_runner_io_methods"
        ) as install_methods,
    ):
        worker._ensure_io_dump_initialized()
        worker._ensure_io_dump_initialized()

    init_io_dump.assert_called_once_with(True)
    install_methods.assert_called_once_with(worker.model_runner)
    register_hooks.assert_called_once_with(model)


def test_nvidia_worker_import_does_not_load_legacy_worker():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class RejectLegacyWorker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {
                    "vllm_fl.worker.worker",
                    "vllm_fl.worker.model_runner",
                }:
                    raise AssertionError(f"NVIDIA imported legacy module: {fullname}")
                return None

        sys.meta_path.insert(0, RejectLegacyWorker())
        from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL
        from vllm.v1.worker.gpu_worker import Worker
        assert issubclass(NvidiaWorkerFL, Worker)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("flagcx", [False, True])
def test_nvidia_compilation_preserves_supported_passes_and_graphs(monkeypatch, flagcx):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config.compilation import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
    )
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
    if flagcx:
        monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    else:
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
    compilation = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        pass_config={
            "fuse_norm_quant": True,
            "fuse_act_quant": True,
            "enable_qk_norm_rope_fusion": True,
            "fuse_rope_kvcache_cat_mla": True,
            "fuse_allreduce_rms": True,
            "fuse_attn_quant": True,
            "eliminate_noops": True,
        },
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", disable_custom_all_reduce=False
        ),
        compilation_config=compilation,
    )
    with patch.object(CudaPlatform, "check_and_update_config"):
        NvidiaPlatformFL.check_and_update_config(config)

    for name in (
        "fuse_norm_quant",
        "fuse_act_quant",
        "enable_qk_norm_rope_fusion",
        "fuse_rope_kvcache_cat_mla",
    ):
        assert getattr(compilation.pass_config, name) is True
    assert compilation.pass_config.fuse_allreduce_rms is not flagcx
    assert compilation.pass_config.fuse_attn_quant is True
    assert compilation.pass_config.eliminate_noops is True
    assert compilation.mode == CompilationMode.VLLM_COMPILE
    assert compilation.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


@pytest.mark.parametrize("use_v2", [False, True])
def test_qsa_patch_runs_after_native_device_selection(monkeypatch, use_v2):
    from types import SimpleNamespace

    import torch

    from vllm.v1.worker.gpu_worker import Worker as NativeGPUWorker

    from vllm_fl.patches import qsa_cache_hint
    from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL

    worker = object.__new__(NvidiaWorkerFL)
    worker.use_v2_model_runner = use_v2
    worker.local_rank = 0
    worker.model_config = SimpleNamespace(
        architecture="Qwen4ExpForConditionalGeneration"
    )
    runner = object()
    calls = []

    def native_init(actual_worker):
        assert actual_worker is worker
        assert actual_worker.use_v2_model_runner is use_v2
        calls.append("native")
        actual_worker.local_rank = 3
        actual_worker.device = torch.device("cuda:3")
        actual_worker.model_runner = runner

    def install(config, device):
        assert calls == ["native"]
        assert config is worker.model_config
        assert device == torch.device("cuda:3")
        assert worker.model_runner is runner
        calls.append("qsa")
        return True

    monkeypatch.setattr(NativeGPUWorker, "init_device", native_init)
    monkeypatch.setattr(qsa_cache_hint, "patch_qsa_cache_hint", install)
    worker.init_device()
    assert calls == ["native", "qsa"]
    assert worker.use_v2_model_runner is use_v2
    assert worker.model_runner is runner
    assert NvidiaWorkerFL.load_model is NativeGPUWorker.load_model


def test_qsa_patch_does_not_run_after_native_device_failure(monkeypatch):
    from unittest.mock import Mock

    from vllm.v1.worker.gpu_worker import Worker as NativeGPUWorker

    from vllm_fl.patches import qsa_cache_hint
    from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL

    worker = object.__new__(NvidiaWorkerFL)

    def fail(_worker):
        raise RuntimeError("native device unavailable")

    install = Mock()
    monkeypatch.setattr(NativeGPUWorker, "init_device", fail)
    monkeypatch.setattr(qsa_cache_hint, "patch_qsa_cache_hint", install)
    with pytest.raises(RuntimeError, match="native device unavailable"):
        worker.init_device()
    install.assert_not_called()
