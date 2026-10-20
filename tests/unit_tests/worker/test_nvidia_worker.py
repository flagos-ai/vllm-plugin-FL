# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA worker/platform contracts for native and empty vLLM builds."""

import pytest

from tests.utils.device_utils import get_backend

# EMPTY builds are intentional NVIDIA targets. Never skip these contracts
# merely because vllm._C_stable_libtorch or the legacy worker is unavailable.
pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA-specific test"
)


@pytest.fixture(scope="module", autouse=True)
def initialize_nvidia_platform():
    # Resolve the real plugin entry point before importing CUDA internals.
    # EMPTY wheels rely on this initialization just as engine startup does.
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


def test_nvidia_platform_selects_target_version_worker_wrapper(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    parallel_config = SimpleNamespace(
        worker_cls=None,
        disable_custom_all_reduce=True,
    )
    vllm_config = SimpleNamespace(
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

    assert parallel_config.worker_cls == "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"
    assert parallel_config.disable_custom_all_reduce is True
    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_keeps_native_cuda_communication(monkeypatch):
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr("vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: True)
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


@pytest.mark.parametrize("native_kernels", [False, True])
def test_nvidia_pin_memory_requires_real_uva_kernel(monkeypatch, native_kernels):
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(
        "vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: native_kernels
    )
    with patch.object(
        CudaPlatform, "is_pin_memory_available", return_value=True
    ) as native:
        assert NvidiaPlatformFL.is_pin_memory_available() is native_kernels
    assert native.call_count == int(native_kernels)


def test_nvidia_empty_disables_custom_allreduce_without_disabling_nccl(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: False
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", disable_custom_all_reduce=False
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=False)
        ),
    )
    with (
        patch.object(CudaPlatform, "check_and_update_config"),
        patch.object(
            CudaPlatform, "get_device_communicator_cls", return_value="native.nccl"
        ),
    ):
        NvidiaPlatformFL.check_and_update_config(config)
        assert config.parallel_config.disable_custom_all_reduce is True
        assert NvidiaPlatformFL.use_custom_allreduce() is False
        assert NvidiaPlatformFL.get_device_communicator_cls() == "native.nccl"
        assert NvidiaPlatformFL.dist_backend == CudaPlatform.dist_backend


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


@pytest.mark.gpu
def test_nvidia_empty_v2_buffers_use_real_device_copies(monkeypatch):
    import torch

    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
    from vllm.v1.worker.gpu.buffer_utils import NonUvaBuffer, UvaBufferPool

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    assert isinstance(current_platform, NvidiaPlatformFL)
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: False
    )
    is_pin_memory_available.cache_clear()
    is_uva_available.cache_clear()
    try:
        assert not is_uva_available()
        pool = UvaBufferPool(2, torch.int32, max_concurrency=2)
        assert all(isinstance(buf, NonUvaBuffer) for buf in pool._uva_bufs)
        first = pool.copy_to_uva([1, 2])
        second = pool.copy_to_uva([3, 4, 5])
        assert first.device.type == second.device.type == "cuda"
        torch.testing.assert_close(first.cpu(), torch.tensor([1, 2], dtype=torch.int32))
        torch.testing.assert_close(
            second.cpu(), torch.tensor([3, 4, 5], dtype=torch.int32)
        )
    finally:
        is_pin_memory_available.cache_clear()
        is_uva_available.cache_clear()


def test_nvidia_empty_filters_unavailable_flash_attention_candidates(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_native_flash_attention", lambda: False
    )
    flash = SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN, priority=0)
    triton = SimpleNamespace(backend=AttentionBackendEnum.TRITON_ATTN, priority=2)
    with patch.object(
        CudaPlatform, "get_valid_backends", return_value=([flash, triton], {})
    ):
        candidates, reasons = NvidiaPlatformFL.get_valid_backends(object(), object())
    assert candidates == [triton]
    assert reasons[AttentionBackendEnum.FLASH_ATTN][0] == 0
    assert "unavailable" in reasons[AttentionBackendEnum.FLASH_ATTN][1][0]


def test_nvidia_empty_vision_backend_excludes_missing_flash_attention(monkeypatch):
    import torch

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_native_flash_attention", lambda: False
    )
    assert (
        AttentionBackendEnum.FLASH_ATTN
        not in NvidiaPlatformFL.get_supported_vit_attn_backends()
    )
    backend = NvidiaPlatformFL.get_vit_attn_backend(64, torch.bfloat16)
    assert backend in {
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    }


@pytest.mark.parametrize("vision", [False, True])
def test_nvidia_empty_rejects_explicit_missing_flash_attention(monkeypatch, vision):
    import torch

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_native_flash_attention", lambda: False
    )
    with pytest.raises(ValueError, match="requires FlashAttention kernels"):
        if vision:
            NvidiaPlatformFL.get_vit_attn_backend(
                64, torch.bfloat16, AttentionBackendEnum.FLASH_ATTN
            )
        else:
            NvidiaPlatformFL.get_attn_backend_cls(
                AttentionBackendEnum.FLASH_ATTN, object()
            )


def test_nvidia_preserves_available_vision_flash_attention(monkeypatch):
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    expected = [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA]
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_native_flash_attention", lambda: True
    )
    with patch.object(
        CudaPlatform, "get_supported_vit_attn_backends", return_value=expected
    ):
        assert NvidiaPlatformFL.get_supported_vit_attn_backends() == expected


@pytest.mark.parametrize("native_kernels", [False, True])
@pytest.mark.parametrize("flagcx", [False, True])
def test_nvidia_compilation_preserves_supported_passes_and_graphs(
    monkeypatch, native_kernels, flagcx
):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config.compilation import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
    )
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(
        "vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: native_kernels
    )
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
        assert getattr(compilation.pass_config, name) is native_kernels
    assert compilation.pass_config.fuse_allreduce_rms is not flagcx
    assert compilation.pass_config.fuse_attn_quant is True
    assert compilation.pass_config.eliminate_noops is True
    assert compilation.mode == CompilationMode.VLLM_COMPILE
    assert compilation.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_nvidia_empty_disables_automatic_mla_fusion(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config.compilation import CompilationConfig
    from vllm.config.vllm import (
        OPTIMIZATION_LEVEL_TO_CONFIG,
        OptimizationLevel,
        VllmConfig,
    )
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(NvidiaPlatformFL, "dist_backend", NvidiaPlatformFL.dist_backend)
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform.has_native_vllm_kernels", lambda: False
    )
    compilation = CompilationConfig(use_inductor_graph_partition=True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", disable_custom_all_reduce=False
        ),
        compilation_config=compilation,
    )
    # Exercise the upstream default setter that runs before the platform hook.
    default = OPTIMIZATION_LEVEL_TO_CONFIG[OptimizationLevel.O2]["compilation_config"][
        "pass_config"
    ]["fuse_rope_kvcache_cat_mla"]
    VllmConfig._set_config_default(
        config, compilation.pass_config, "fuse_rope_kvcache_cat_mla", default
    )
    assert compilation.pass_config.fuse_rope_kvcache_cat_mla is True
    with patch.object(CudaPlatform, "check_and_update_config"):
        NvidiaPlatformFL.check_and_update_config(config)
    assert compilation.pass_config.fuse_rope_kvcache_cat_mla is False
    assert compilation.use_inductor_graph_partition is True


@pytest.mark.parametrize("cache_writer", [False, True])
def test_nvidia_filters_backends_requiring_cache_kernels(monkeypatch, cache_writer):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_cuda_cache_writer", lambda: cache_writer
    )
    monkeypatch.setattr(
        "vllm_fl.nvidia_platform._has_native_flash_attention", lambda: True
    )
    candidates = [
        SimpleNamespace(backend=AttentionBackendEnum[name], priority=index)
        for index, name in enumerate(
            ["FLASHINFER", "FLEX_ATTENTION", "TRITON_ATTN", "TRITON_FLASHINFER"]
        )
        if name in AttentionBackendEnum.__members__
    ]
    with patch.object(
        CudaPlatform, "get_valid_backends", return_value=(candidates, {})
    ):
        available, reasons = NvidiaPlatformFL.get_valid_backends(object(), object())
    assert available == (candidates if cache_writer else candidates[2:])
    assert len(reasons) == (0 if cache_writer else 2)


@pytest.mark.parametrize("backend_name", ["FLASHINFER", "FLEX_ATTENTION"])
def test_nvidia_empty_rejects_explicit_native_cache_backend(monkeypatch, backend_name):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setattr("vllm_fl.nvidia_platform._has_cuda_cache_writer", lambda: False)
    with pytest.raises(ValueError, match="CUDA cache kernels"):
        NvidiaPlatformFL.get_attn_backend_cls(
            AttentionBackendEnum[backend_name], object()
        )


@pytest.mark.parametrize("load_dummy_weights", [False, True])
def test_nvidia_worker_installs_model_kernels_before_native_loading(
    monkeypatch, load_dummy_weights
):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker as NativeGPUWorker

    from vllm_fl.ops import nvidia_dsv41, nvidia_qsa
    from vllm_fl.worker.nvidia_worker import NvidiaWorkerFL

    worker = object.__new__(NvidiaWorkerFL)
    config = SimpleNamespace()
    worker.vllm_config = config
    calls = []

    def install(actual_config):
        assert actual_config is config
        calls.append("install")

    def install_qsa(actual_config):
        assert actual_config is config
        calls.append("qsa")

    def load(actual_worker, *, load_dummy_weights):
        assert actual_worker is worker
        assert calls == ["install", "qsa"]
        calls.append(load_dummy_weights)

    monkeypatch.setattr(nvidia_dsv41, "install_dsv41_empty_adapters", install)
    monkeypatch.setattr(nvidia_qsa, "install_nvidia_qsa", install_qsa)
    monkeypatch.setattr(NativeGPUWorker, "load_model", load)
    worker.load_model(load_dummy_weights=load_dummy_weights)
    assert calls == ["install", "qsa", load_dummy_weights]
