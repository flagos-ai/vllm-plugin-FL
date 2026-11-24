"""Coverage tests for ModelRunnerFL.load_model (model_runner.py @ d12e9d9e).

Only ``load_model`` is exercised.  The real module is imported and a bare
instance is created with ``object.__new__``; every loader, model, drafter and
graph-wrapper dependency is replaced with a local double.  No weights are
read and no device work is performed.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.config import CompilationMode, CUDAGraphMode
from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL


class _MemProfiler:
    def __init__(self, consumed):
        self.consumed_memory = consumed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOOM(Exception):
    pass


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _compilation(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE):
    return SimpleNamespace(
        mode=mode,
        cudagraph_mode=cudagraph_mode,
        init_backend=MagicMock(),
    )


def _make_runner(model=None, compilation=None, **overrides):
    model = model if model is not None else SimpleNamespace()
    comp = compilation if compilation is not None else _compilation()
    attrs = dict(
        model_config=SimpleNamespace(
            model="test-model", enforce_eager=False, multimodal_config=None
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=False, enable_elastic_ep=False, use_ubatching=False
        ),
        load_config=SimpleNamespace(load_format="auto"),
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
        compilation_config=comp,
        device="dev",
        graph_runtime=None,
        lora_config=None,
        eplb_state=None,
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
        model=model,
        _moe_model=None,
    )
    attrs.update(overrides)
    runner = new_runner(**attrs)
    runner.get_model = lambda: runner.model
    runner._setup_eagle3_aux_hidden_state_outputs = MagicMock()
    return runner


def _patch_common(monkeypatch, loader=None, moe=None, pruning=False):
    monkeypatch.setattr(mr, "logger", MagicMock())
    monkeypatch.setattr(mr, "init_io_dump_from_env", MagicMock())
    monkeypatch.setattr(mr, "format_gib", lambda value: f"{value}GiB")
    monkeypatch.setattr(mr, "warmup_cached_ops", MagicMock())
    monkeypatch.setattr(mr, "_prepare_comm_buffer", MagicMock())
    monkeypatch.setattr(mr, "register_io_module_hooks", MagicMock())
    monkeypatch.setattr(
        mr, "supports_multimodal_pruning", MagicMock(return_value=pruning)
    )
    monkeypatch.setattr(
        mr, "get_model_loader", MagicMock(return_value=loader)
    )
    monkeypatch.setattr(mr, "DeviceMemoryProfiler", lambda: _MemProfiler(7))
    monkeypatch.setattr(
        mr, "time", SimpleNamespace(perf_counter=lambda: 2.5)
    )
    monkeypatch.setattr(
        mr, "get_mixture_of_experts_model", MagicMock(return_value=moe)
    )
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(
            device_type="cuda",
            get_static_graph_wrapper_cls=lambda: "static-wrapper",
        ),
    )
    monkeypatch.setattr(mr, "is_breakable_cudagraph_enabled", lambda: False)
    offloader = SimpleNamespace(post_init=MagicMock())
    monkeypatch.setattr(mr, "get_offloader", lambda: offloader)
    return offloader


def test_load_model_basic_success(monkeypatch):
    model = SimpleNamespace(name="loaded")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    runner = _make_runner()
    offloader = _patch_common(monkeypatch, loader=loader)

    MR.load_model(runner)

    mr.init_io_dump_from_env.assert_called_once_with(False)
    loader.load_model.assert_called_once_with(
        vllm_config=runner.vllm_config, model_config=runner.model_config
    )
    assert runner.model is model
    assert runner.model_memory_usage == 7
    mr.warmup_cached_ops.assert_called_once_with()
    mr._prepare_comm_buffer.assert_called_once_with(model)
    mr.register_io_module_hooks.assert_called_once_with(model)
    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is False
    assert runner._moe_model is None
    runner._setup_eagle3_aux_hidden_state_outputs.assert_called_once_with()
    offloader.post_init.assert_called_once_with()
    assert mr.logger.info_once.called


def test_load_model_dummy_weights_with_lora_skips_comm_hooks(monkeypatch):
    base = SimpleNamespace(name="base")
    lora = SimpleNamespace(name="lora")
    loader = SimpleNamespace(load_model=MagicMock(return_value=base))
    runner = _make_runner(
        model_config=SimpleNamespace(
            model="test-model", enforce_eager=True, multimodal_config=None
        ),
        lora_config=object(),
    )
    runner.load_lora_model = MagicMock(return_value=lora)
    offloader = _patch_common(monkeypatch, loader=loader)

    MR.load_model(runner, load_dummy_weights=True)

    assert runner.load_config.load_format == "dummy"
    mr.init_io_dump_from_env.assert_called_once_with(True)
    runner.load_lora_model.assert_called_once_with(base, runner.vllm_config, "dev")
    assert runner.model is lora
    mr._prepare_comm_buffer.assert_not_called()
    mr.register_io_module_hooks.assert_not_called()
    mr.warmup_cached_ops.assert_called_once_with()
    offloader.post_init.assert_called_once_with()


def test_load_model_drafter_loads_and_sets_eplb(monkeypatch):
    model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(name="drafter")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    drafter = SimpleNamespace(
        model=drafter_model,
        load_model=MagicMock(),
        set_eplb_state=MagicMock(),
    )
    comp = _compilation()
    runner = _make_runner(
        parallel_config=SimpleNamespace(
            enable_eplb=True, enable_elastic_ep=False, use_ubatching=False
        ),
        drafter=drafter,
        vllm_config=SimpleNamespace(
            compilation_config=comp,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(model="draft-model")
            ),
        ),
        compilation_config=comp,
    )
    moe_drafter = object()
    moe_target = object()

    def pick(target):
        if target is drafter_model:
            return moe_drafter
        if target is model:
            return moe_target
        return None

    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(mr, "get_mixture_of_experts_model", pick)
    state = SimpleNamespace(
        add_model=MagicMock(), is_async=True, start_async_loop=MagicMock()
    )
    eplb_cls = MagicMock(return_value=state)
    monkeypatch.setattr(mr, "EplbState", eplb_cls)

    MR.load_model(runner)

    eplb_cls.assert_called_once_with(runner.parallel_config, "dev")
    drafter.load_model.assert_called_once_with(model)
    assert state.add_model.call_count == 2
    state.add_model.assert_any_call(
        moe_drafter, runner.vllm_config.speculative_config.draft_model_config
    )
    state.add_model.assert_any_call(moe_target, runner.model_config)
    drafter.set_eplb_state.assert_called_once_with(state)
    state.start_async_loop.assert_called_once_with()
    assert runner._moe_model is moe_target


def test_load_model_drafter_without_moe_or_eplb(monkeypatch):
    model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(name="drafter")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    drafter = SimpleNamespace(model=drafter_model)
    runner = _make_runner(drafter=drafter)
    _patch_common(monkeypatch, loader=loader, moe=None)

    MR.load_model(runner)

    assert runner.model is model
    assert not hasattr(drafter, "load_model")
    assert not hasattr(drafter, "set_eplb_state")
    assert runner._moe_model is None
    mr.logger.info_once.assert_any_call("Loading drafter model...")


def test_load_model_multimodal_flags(monkeypatch):
    model = SimpleNamespace(requires_sequential_video_encoding=object())
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    mm_config = SimpleNamespace(
        is_multimodal_pruning_enabled=MagicMock(return_value=True)
    )
    runner = _make_runner(
        model_config=SimpleNamespace(
            model="test-model", enforce_eager=False, multimodal_config=mm_config
        )
    )
    _patch_common(monkeypatch, loader=loader, pruning=True)

    MR.load_model(runner)

    assert runner.is_multimodal_pruning_enabled is True
    assert runner.requires_sequential_video_encoding is True
    mm_config.is_multimodal_pruning_enabled.assert_called_once_with()


def test_load_model_out_of_memory_is_logged_and_reraised(monkeypatch):
    monkeypatch.setattr(
        mr.torch, "OutOfMemoryError", _FakeOOM, raising=False
    )
    loader = SimpleNamespace(
        load_model=MagicMock(side_effect=_FakeOOM("boom"))
    )
    runner = _make_runner()
    offloader = _patch_common(monkeypatch, loader=loader)

    with pytest.raises(_FakeOOM):
        MR.load_model(runner)

    message = mr.logger.error.call_args[0][0]
    assert "not enough GPU memory" in message
    assert "boom" in message
    mr.warmup_cached_ops.assert_not_called()
    offloader.post_init.assert_not_called()
    assert not hasattr(runner, "model_memory_usage")


def test_load_model_musa_dispatch_warmup(monkeypatch):
    from vllm_fl import dispatch

    dispatch_calls = []
    monkeypatch.setattr(
        dispatch, "warmup_cached_ops", lambda: dispatch_calls.append(1)
    )
    model = SimpleNamespace(name="loaded")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(mode=object())
    runner = _make_runner(
        compilation=comp,
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(
            device_type="musa", get_static_graph_wrapper_cls=lambda: "cls"
        ),
    )
    import torch._dynamo.config as dynamo_config

    ignored = set()
    monkeypatch.setattr(
        dynamo_config, "ignore_logger_methods", ignored, raising=False
    )

    MR.load_model(runner)

    mr.warmup_cached_ops.assert_called_once_with()
    assert dispatch_calls == [1]
    assert runner.model is model
    import logging

    assert logging.Logger.debug in ignored


def test_load_model_npu_dispatch_warmup(monkeypatch):
    from vllm_fl import dispatch

    dispatch_calls = []
    monkeypatch.setattr(
        dispatch, "warmup_cached_ops", lambda: dispatch_calls.append(1)
    )
    loader = SimpleNamespace(
        load_model=MagicMock(return_value=SimpleNamespace(name="loaded"))
    )
    comp = _compilation(mode=object())
    runner = _make_runner(
        compilation=comp,
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(
            device_type="npu", get_static_graph_wrapper_cls=lambda: "cls"
        ),
    )

    MR.load_model(runner)

    mr.warmup_cached_ops.assert_called_once_with()
    assert dispatch_calls == [1]


def test_load_model_stock_torch_compile_returns_early(monkeypatch):
    import vllm.env_override as env_override

    patch_calls = []
    monkeypatch.setattr(
        env_override,
        "_apply_constrain_to_fx_strides_patch",
        lambda: patch_calls.append(1),
    )
    backend = object()
    comp = SimpleNamespace(
        mode=CompilationMode.STOCK_TORCH_COMPILE,
        cudagraph_mode=CUDAGraphMode.NONE,
        init_backend=MagicMock(return_value=backend),
    )
    model = SimpleNamespace(compile=MagicMock())
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    runner = _make_runner(
        compilation=comp,
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    offloader = _patch_common(monkeypatch, loader=loader)
    counter = SimpleNamespace(stock_torch_compile_count=0)
    monkeypatch.setattr(mr, "compilation_counter", counter)

    MR.load_model(runner)

    assert patch_calls == [1]
    comp.init_backend.assert_called_once_with(runner.vllm_config)
    model.compile.assert_called_once_with(fullgraph=True, backend=backend)
    assert counter.stock_torch_compile_count == 1
    offloader.post_init.assert_not_called()


def test_load_model_breakable_cudagraph_wraps_model_and_drafter(monkeypatch):
    model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(name="drafter")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(cudagraph_mode=CUDAGraphMode.FULL)
    runner = _make_runner(
        compilation=comp,
        drafter=SimpleNamespace(model=drafter_model),
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    wrapper_cls = MagicMock(
        side_effect=lambda mod, cfg: SimpleNamespace(wrapped=mod, config=cfg)
    )
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(mr, "is_breakable_cudagraph_enabled", lambda: True)
    monkeypatch.setattr(mr, "BreakableCUDAGraphWrapper", wrapper_cls)

    MR.load_model(runner)

    assert wrapper_cls.call_count == 2
    wrapper_cls.assert_any_call(model, runner.vllm_config)
    wrapper_cls.assert_any_call(drafter_model, runner.vllm_config)
    assert runner.model.wrapped is model
    assert runner.drafter.model.wrapped is drafter_model


def test_load_model_full_cudagraph_static_wrapper(monkeypatch):
    model = SimpleNamespace(name="target")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(cudagraph_mode=CUDAGraphMode.FULL)
    runner = _make_runner(
        compilation=comp,
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    wrapper_cls = MagicMock(return_value="wrapped")
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(
        mr, "resolve_obj_by_qualname", lambda qualname: wrapper_cls
    )

    MR.load_model(runner)

    wrapper_cls.assert_called_once_with(
        model, runner.vllm_config, runtime_mode=CUDAGraphMode.FULL
    )
    assert runner.model == "wrapped"


def test_load_model_full_cudagraph_with_graph_runtime(monkeypatch):
    model = SimpleNamespace(name="target")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(cudagraph_mode=CUDAGraphMode.FULL)
    runner = _make_runner(
        compilation=comp,
        graph_runtime="runtime",
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    wrapper_cls = MagicMock(return_value="wrapped")
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(
        mr, "resolve_obj_by_qualname", lambda qualname: wrapper_cls
    )

    MR.load_model(runner)

    wrapper_cls.assert_called_once_with(
        model,
        runner.vllm_config,
        runtime_mode=CUDAGraphMode.FULL,
        graph_runtime="runtime",
    )


def test_load_model_ubatching_full_wrapper(monkeypatch):
    model = SimpleNamespace(name="target")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(cudagraph_mode=CUDAGraphMode.FULL)
    runner = _make_runner(
        compilation=comp,
        parallel_config=SimpleNamespace(
            enable_eplb=False, enable_elastic_ep=False, use_ubatching=True
        ),
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    wrapper_cls = MagicMock(return_value="ubatch")
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(mr, "UBatchWrapper", wrapper_cls)

    MR.load_model(runner)

    wrapper_cls.assert_called_once_with(
        model, runner.vllm_config, CUDAGraphMode.FULL, "dev"
    )
    assert runner.model == "ubatch"


def test_load_model_ubatching_none_wrapper(monkeypatch):
    model = SimpleNamespace(name="target")
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    comp = _compilation(cudagraph_mode=CUDAGraphMode.NONE)
    runner = _make_runner(
        compilation=comp,
        parallel_config=SimpleNamespace(
            enable_eplb=False, enable_elastic_ep=False, use_ubatching=True
        ),
        vllm_config=SimpleNamespace(
            compilation_config=comp, speculative_config=None
        ),
    )
    wrapper_cls = MagicMock(return_value="ubatch-none")
    _patch_common(monkeypatch, loader=loader)
    monkeypatch.setattr(mr, "UBatchWrapper", wrapper_cls)

    MR.load_model(runner)

    wrapper_cls.assert_called_once_with(
        model, runner.vllm_config, CUDAGraphMode.NONE, "dev"
    )
    assert runner.model == "ubatch-none"
