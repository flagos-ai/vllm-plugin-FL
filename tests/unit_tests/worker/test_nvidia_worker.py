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

    from vllm_fl.platform import PlatformFL

    assert isinstance(current_platform, PlatformFL)


def test_nvidia_platform_keeps_native_cuda_semantics():
    from vllm.platforms import PlatformEnum
    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.platform import PlatformFL

    assert issubclass(PlatformFL, CudaPlatform)
    assert PlatformFL._enum == PlatformEnum.CUDA
    platform = PlatformFL()
    assert platform.is_cuda()
    assert not platform.is_out_of_tree()


@pytest.mark.parametrize("use_v2", [False, True])
def test_nvidia_platform_selects_target_version_worker_wrapper(monkeypatch, use_v2):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.platform import PlatformFL

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
        state.setattr(PlatformFL, "dist_backend", PlatformFL.dist_backend)
        state.delenv("FLAGCX_PATH", raising=False)
        with patch.object(CudaPlatform, "check_and_update_config") as native_update:
            PlatformFL.check_and_update_config(vllm_config)

        assert PlatformFL.dist_backend == CudaPlatform.dist_backend

    assert vllm_config.use_v2_model_runner is use_v2
    assert parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
    assert parallel_config.disable_custom_all_reduce is True
    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_keeps_native_cuda_communication(monkeypatch):
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.platform import PlatformFL

    with monkeypatch.context() as state:
        state.setattr(PlatformFL, "dist_backend", PlatformFL.dist_backend)
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
            communicator = PlatformFL.get_device_communicator_cls()
            use_custom_allreduce = PlatformFL.use_custom_allreduce()

        assert PlatformFL.dist_backend == CudaPlatform.dist_backend

    assert communicator == "native.cuda.Communicator"
    assert use_custom_allreduce is True
    native_communicator.assert_called_once_with()
    native_custom_allreduce.assert_called_once_with()


def test_nvidia_platform_uses_flagcx_when_configured(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.platform import PlatformFL

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
        state.setattr(PlatformFL, "dist_backend", PlatformFL.dist_backend)
        state.setenv("FLAGCX_PATH", "/opt/flagcx")
        with patch.object(CudaPlatform, "check_and_update_config") as native_update:
            PlatformFL.check_and_update_config(vllm_config)

        assert PlatformFL.dist_backend == "flagcx"
        assert parallel_config.disable_custom_all_reduce is True
        assert (
            PlatformFL.get_device_communicator_cls()
            == "vllm_fl.distributed.communicator.CommunicatorFL"
        )
        assert PlatformFL.use_custom_allreduce() is False

    native_update.assert_called_once_with(vllm_config)


def test_nvidia_platform_uses_native_attention_by_default(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    from vllm_fl.platform import PlatformFL

    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    selector = SimpleNamespace(use_mla=False, use_sparse=False)

    with patch.object(
        CudaPlatform,
        "get_attn_backend_cls",
        return_value="native.attention.Backend",
    ) as native_select:
        result = PlatformFL.get_attn_backend_cls(None, selector, 32)

    assert result == "native.attention.Backend"
    native_select.assert_called_once_with(None, selector, 32)


def test_nvidia_platform_honors_explicit_flaggems_attention(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.platform import PlatformFL

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
        result = PlatformFL.get_attn_backend_cls(None, selector, 32)

    assert result == backend_path
    flaggems_select.assert_called_once_with(use_mla=False, use_sparse=False)


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

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "dist_backend", PlatformFL.dist_backend)
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
        PlatformFL.check_and_update_config(config)

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


def test_common_cuda_metadata_queries_do_not_initialize_cuda(monkeypatch):
    import torch

    from vllm.platforms.cuda import NvmlCudaPlatform

    from vllm_fl.platform import PlatformFL

    if not issubclass(PlatformFL, NvmlCudaPlatform):
        pytest.skip("Requires the native NVML platform")

    def reject_cuda_initialization():
        raise AssertionError("Metadata queries must preserve native NVML lookup")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda_initialization)
    assert PlatformFL.get_device_capability(0).major > 0
    assert PlatformFL.get_device_name(0)
    assert PlatformFL.get_device_total_memory(0) > 0
