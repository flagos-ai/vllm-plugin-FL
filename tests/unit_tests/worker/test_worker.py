# Copyright (c) 2025 BAAI. All rights reserved.

"""Contracts for the multi-vendor worker and its runtime policies."""

from types import SimpleNamespace

import pytest


def _sequence_parallel_config(enabled, tp, *, partition=False, splitting_ops=()):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enabled),
            use_inductor_graph_partition=partition,
            splitting_ops=list(splitting_ops),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.mark.parametrize(
    "enabled,tp,tokens,partition,expected",
    [
        (False, 2, 3, False, False),
        (True, 1, 3, False, False),
        (True, 2, 4, False, True),
        (True, 2, 4, True, True),
    ],
)
def test_legacy_pipeline_residual_layout_when_upstream_helper_is_absent(
    monkeypatch, enabled, tp, tokens, partition, expected
):
    from vllm.v1.worker import utils as worker_utils

    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    monkeypatch.delattr(worker_utils, "is_residual_scattered_for_sp", raising=False)
    config = _sequence_parallel_config(enabled, tp, partition=partition)
    assert is_residual_scattered_for_sp(config, tokens) is expected


@pytest.mark.parametrize("tokens,splitting_ops", [(3, ()), (4, ("split",))])
def test_legacy_pipeline_residual_layout_rejects_invalid_sp_configuration(
    monkeypatch, tokens, splitting_ops
):
    from vllm.v1.worker import utils as worker_utils

    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    monkeypatch.delattr(worker_utils, "is_residual_scattered_for_sp", raising=False)
    config = _sequence_parallel_config(True, 2, splitting_ops=splitting_ops)
    with pytest.raises(AssertionError, match="Sequence parallelism requires"):
        is_residual_scattered_for_sp(config, tokens)


def test_pipeline_residual_layout_preserves_native_helper_errors(monkeypatch):
    from vllm.v1.worker import utils as worker_utils

    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    error = RuntimeError("native layout failure")

    def native(config, tokens):
        raise error

    monkeypatch.setattr(
        worker_utils, "is_residual_scattered_for_sp", native, raising=False
    )
    with pytest.raises(RuntimeError) as caught:
        is_residual_scattered_for_sp(_sequence_parallel_config(True, 2), 4)
    assert caught.value is error


def test_pipeline_residual_layout_preserves_native_helper_result(monkeypatch):
    from vllm.v1.worker import utils as worker_utils

    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    calls = []

    def native(config, tokens):
        calls.append((config, tokens))
        return False

    monkeypatch.setattr(
        worker_utils, "is_residual_scattered_for_sp", native, raising=False
    )
    config = _sequence_parallel_config(True, 2)
    assert is_residual_scattered_for_sp(config, 4) is False
    assert calls == [(config, 4)]


@pytest.mark.parametrize(
    "device_type,vllm_version,use_native",
    [
        ("gcu", "0.28.0+empty", False),
        ("gcu", "0.32.0+empty", True),
        ("npu", "0.32.0", False),
        ("cuda", "0.32.0", False),
    ],
)
def test_model_runner_selection_keeps_other_backends_unchanged(
    monkeypatch, device_type, vllm_version, use_native
):
    import importlib.metadata
    import sys
    from types import ModuleType

    from vllm_fl.worker.runtime import get_fl_model_runner_cls

    legacy = type("LegacyRunner", (), {})
    native = type("NativeRunner", (), {})
    for path, symbol, cls in [
        ("vllm_fl.worker.model_runner", "ModelRunnerFL", legacy),
        ("vllm_fl.worker.enflame_model_runner", "EnflameModelRunner", native),
    ]:
        module = ModuleType(path)
        setattr(module, symbol, cls)
        monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: vllm_version)

    assert get_fl_model_runner_cls(device_type) is (native if use_native else legacy)


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
    assert "get_fl_model_runner_cls(self.device.type)" in source


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


@pytest.mark.parametrize("mode", ["NONE", "VLLM_COMPILE"])
def test_gcu_ir_defaults_only_select_registered_native_providers(monkeypatch, mode):
    from dataclasses import fields

    from vllm.config.compilation import CompilationMode
    from vllm.ir.op import IrOp

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "gcu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            backend="inductor", mode=getattr(CompilationMode, mode)
        )
    )
    priority = PlatformFL.get_default_ir_op_priority(config)
    for field in fields(priority):
        providers = getattr(priority, field.name)
        assert providers == ["native"]
        assert all(
            provider in IrOp.registry[field.name].impls for provider in providers
        )


def test_batch_invariant_module_resolves_installed_upstream_api():
    from importlib.metadata import version

    from packaging.version import Version

    from vllm_fl.utils import get_batch_invariant_module

    module = get_batch_invariant_module()
    parent = (
        "determinism" if Version(version("vllm")) >= Version("0.32.0") else "layers"
    )
    assert module.__name__ == f"vllm.model_executor.{parent}.batch_invariant"
    assert callable(module.init_batch_invariance)
    assert isinstance(module._batch_invariant_MODE, bool)


@pytest.mark.parametrize("legacy_flag", [None, False, True])
def test_worker_initializes_cache_with_legacy_or_relocated_routing_config(
    monkeypatch, legacy_flag
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm_fl.worker import worker as module

    worker = object.__new__(module.WorkerFL)
    fields = (
        {} if legacy_flag is None else {"enable_return_routed_experts": legacy_flag}
    )
    worker.model_config = SimpleNamespace(**fields)
    worker.cache_config = SimpleNamespace()
    worker.vllm_config = SimpleNamespace()
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=Mock(), init_routed_experts_capturer=Mock()
    )
    worker._maybe_get_memory_pool_context = lambda **kwargs: nullcontext()
    transfer = Mock()
    monkeypatch.setattr(module, "ensure_kv_transfer_initialized", transfer)
    config = SimpleNamespace(num_blocks=7, needs_kv_cache_zeroing=False)
    worker.initialize_from_config(config)
    transfer.assert_called_once_with(worker.vllm_config, config)
    worker.model_runner.initialize_kv_cache.assert_called_once_with(config)
    assert worker.cache_config.num_gpu_blocks == 7
    assert worker.model_runner.init_routed_experts_capturer.call_count == int(
        legacy_flag is True
    )


def test_gcu_rejects_v2_auxiliary_routing_before_model_initialization(monkeypatch):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "gcu")
    config = SimpleNamespace(
        use_v2_model_runner=False,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True),
    )
    with pytest.raises(ValueError, match="routed-expert auxiliary outputs"):
        PlatformFL.check_and_update_config(config)
