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
def test_pipeline_residual_layout(enabled, tp, tokens, partition, expected):
    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    config = _sequence_parallel_config(enabled, tp, partition=partition)
    assert is_residual_scattered_for_sp(config, tokens) is expected


@pytest.mark.parametrize("tokens,splitting_ops", [(3, ()), (4, ("split",))])
def test_pipeline_residual_layout_rejects_invalid_sp_configuration(
    tokens, splitting_ops
):
    from vllm_fl.worker.runtime import is_residual_scattered_for_sp

    config = _sequence_parallel_config(True, 2, splitting_ops=splitting_ops)
    with pytest.raises(AssertionError, match="Sequence parallelism requires"):
        is_residual_scattered_for_sp(config, tokens)


@pytest.mark.parametrize("device_type", ["gcu", "npu", "cuda", "musa"])
def test_common_runner_is_owned_by_fl_for_every_platform(monkeypatch, device_type):
    import vllm.platforms
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_fl.worker.model_runner import ModelRunnerFL

    monkeypatch.setattr(vllm.platforms.current_platform, "device_type", device_type)
    assert ModelRunnerFL is not GPUModelRunner
    assert not issubclass(ModelRunnerFL, GPUModelRunner)
    for name in (
        "__init__",
        "load_model",
        "execute_model",
        "sample_tokens",
        "capture_model",
    ):
        assert getattr(ModelRunnerFL, name).__module__ == "vllm_fl.worker.model_runner"


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
    assert "ModelRunnerFL(" in source


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
    monkeypatch.setattr(PlatformFL, "vendor_name", "enflame")
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


def test_worker_initializes_cache_with_current_runner_contract(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm_fl.worker import worker as module

    worker = object.__new__(module.WorkerFL)
    worker.model_config = SimpleNamespace()
    worker.cache_config = SimpleNamespace()
    worker.vllm_config = SimpleNamespace()
    worker.model_runner = SimpleNamespace(initialize_kv_cache=Mock())
    worker._maybe_get_memory_pool_context = lambda **kwargs: nullcontext()
    transfer = Mock()
    monkeypatch.setattr(module, "ensure_kv_transfer_initialized", transfer)
    config = SimpleNamespace(num_blocks=7, needs_kv_cache_zeroing=False)
    worker.initialize_from_config(config)
    transfer.assert_called_once_with(worker.vllm_config, config)
    worker.model_runner.initialize_kv_cache.assert_called_once_with(config)
    assert worker.cache_config.num_gpu_blocks == 7


def test_gcu_rejects_v2_auxiliary_routing_before_model_initialization(monkeypatch):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "gcu")
    monkeypatch.setattr(PlatformFL, "vendor_name", "enflame")
    config = SimpleNamespace(
        use_v2_model_runner=False,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True),
    )
    with pytest.raises(ValueError, match="routed-expert auxiliary outputs"):
        PlatformFL.check_and_update_config(config)


def test_native_runner_io_bridge_replaces_only_inference_mode():
    import torch

    from vllm_fl.dispatch.io_common import set_io_active
    from vllm_fl.worker.runtime import _install_native_runner_io_methods

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


@pytest.mark.parametrize("io_enabled", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
def test_explicit_v2_preparation_preserves_shared_io_hooks(
    monkeypatch, io_enabled, dummy
):
    import vllm_fl.dispatch.io_dumper as io_dumper
    import vllm_fl.worker.runtime as runtime

    events = []
    model = object()
    runner = SimpleNamespace(model=model, get_model=lambda: model)
    monkeypatch.setattr(
        io_dumper, "init_io_dump_from_env", lambda eager: events.append(("init", eager))
    )
    monkeypatch.setattr(io_dumper, "is_dump_enabled", lambda: io_enabled)
    monkeypatch.setattr(
        io_dumper,
        "register_io_module_hooks",
        lambda value: events.append(("hooks", value)),
    )
    monkeypatch.setattr(
        runtime,
        "_install_native_runner_io_methods",
        lambda value, names: events.append(("methods", value, names)),
    )
    runtime.prepare_native_model_runner(runner, True, dummy)
    expected = [("init", True)]
    if io_enabled:
        expected.append(
            (
                "methods",
                runner,
                ("execute_model", "sample_tokens", "pool"),
            )
        )
        if not dummy:
            expected.append(("hooks", model))
    assert runner.model is model
    assert events == expected


@pytest.mark.parametrize("method_name", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("has_output", [False, True])
def test_shared_native_io_hooks_advance_once_for_completed_output(
    monkeypatch, method_name, has_output
):
    import torch

    import vllm_fl.dispatch.io_dumper as io_dumper
    from vllm_fl.worker.runtime import _install_native_runner_io_methods

    class NativeRunner:
        @torch.inference_mode()
        def execute_model(self, output):
            return output

        @torch.inference_mode()
        def sample_tokens(self, output):
            return output

    events = []
    monkeypatch.setattr(io_dumper, "advance_io_step", lambda: events.append("step"))
    runner = NativeRunner()
    _install_native_runner_io_methods(runner)
    _install_native_runner_io_methods(runner)
    output = object() if has_output else None
    assert getattr(runner, method_name)(output) is output
    assert events == (["step"] if has_output else [])


def test_worker_config_update_keeps_runner_and_shared_config_in_sync():
    from vllm.config import LoadConfig

    from vllm_fl.worker.worker import WorkerFL

    worker = object.__new__(WorkerFL)
    worker.load_config = LoadConfig()
    worker.vllm_config = SimpleNamespace(load_config=worker.load_config)
    worker.model_runner = SimpleNamespace(load_config=worker.load_config)

    worker.update_config({"load_config": {"download_dir": "/tmp/fl-config-test"}})

    assert worker.load_config.download_dir == "/tmp/fl-config-test"
    assert worker.load_config is worker.vllm_config.load_config
    assert worker.load_config is worker.model_runner.load_config
    with pytest.raises(ValueError, match="cache_config"):
        worker.update_config({"cache_config": {}})


@pytest.mark.parametrize("wrapper_kind", ["fl", "native"])
def test_common_runner_unwraps_graph_models(monkeypatch, wrapper_kind):
    import torch

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode

    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.worker.model_runner import ModelRunnerFL

    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_global_graph_pool", lambda: None
    )
    model = torch.nn.Linear(2, 2)
    wrapper_cls = GraphWrapper if wrapper_kind == "fl" else CUDAGraphWrapper
    wrapper = wrapper_cls(
        model, SimpleNamespace(compilation_config=SimpleNamespace()), CUDAGraphMode.FULL
    )
    runner = ModelRunnerFL.__new__(ModelRunnerFL)
    runner.model = wrapper
    runner.drafter = SimpleNamespace(model=wrapper)
    assert runner.get_model() is model
    assert runner.get_draft_model() is model


@pytest.mark.parametrize("wrapper_kind", ["fl", "native", "plain", "manager"])
def test_elastic_ep_releases_graphs_before_native_cleanup(monkeypatch, wrapper_kind):
    from contextlib import nullcontext

    import torch

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.distributed.elastic_ep import elastic_execute

    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.worker.worker import ElasticEPScalingExecutorFL

    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_global_graph_pool", lambda: None
    )
    model = torch.nn.Linear(2, 2)
    runner = SimpleNamespace(get_model=lambda: model)
    if wrapper_kind in {"fl", "native"}:
        wrapper_cls = GraphWrapper if wrapper_kind == "fl" else CUDAGraphWrapper
        runner.model = wrapper_cls(
            model,
            SimpleNamespace(compilation_config=SimpleNamespace()),
            CUDAGraphMode.FULL,
        )
        runner.model.concrete_cudagraph_entries["capture"] = object()
    elif wrapper_kind == "plain":
        runner.model = model
    events = []
    if wrapper_kind == "manager":
        runner.cudagraph_manager = SimpleNamespace(
            release_graphs=lambda: events.append("manager")
        )

    def check_captures_released():
        if wrapper_kind in {"fl", "native"}:
            assert runner.model.concrete_cudagraph_entries == {}
        events.append("reset")

    monkeypatch.setattr(torch.compiler, "reset", check_captures_released)
    monkeypatch.setattr(
        elastic_execute, "set_current_vllm_config", lambda _: nullcontext()
    )
    monkeypatch.setattr(
        elastic_execute, "reset_compile_wrapper", lambda raw: events.append(raw)
    )
    monkeypatch.setattr(elastic_execute.gc, "collect", lambda: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: events.append("empty")
    )
    executor = ElasticEPScalingExecutorFL.__new__(ElasticEPScalingExecutorFL)
    worker = SimpleNamespace(model_runner=runner, vllm_config=object())
    executor.worker_ref = lambda: worker
    executor._release_cuda_graphs()
    expected = (["manager"] if wrapper_kind == "manager" else []) + [
        "reset",
        model,
        "sync",
        "empty",
    ]
    assert events == expected
