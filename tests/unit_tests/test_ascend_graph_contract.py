# SPDX-License-Identifier: Apache-2.0
"""Graph contracts with runtime stubs; importing this test requires no NPU."""

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_fl"


def _ascend_phase_helpers():
    tree = ast.parse(
        (ROOT / "dispatch/backends/vendor/ascend/impl/attention.py").read_text()
    )
    names = {
        "AscendAttentionState",
        "classify_ascend_attention_state",
        "get_ascend_dummy_attention_state",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    # Keep Python 3.9 CPU CI independent of runtime annotation imports and NPU
    # registration while executing the real enum and function definitions.
    future = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    namespace = {"Enum": Enum}
    exec(compile(module, "ascend-phases", "exec"), namespace)
    return NS(**namespace)


@pytest.mark.parametrize(
    "computed,scheduled,valid,chunked,method,expected",
    [
        ([0], [1], [1], False, None, "PrefillNoCache"),
        ([0, 0], [1, 4], [1, 4], True, "mtp", "PrefillNoCache"),
        ([5], [1], [1], True, None, "DecodeOnly"),
        ([5], [1], [1], False, "mtp", "SpecDecoding"),
        ([5], [1], [1], False, "eagle3", "DecodeOnly"),
        ([5], [3], [1], False, "mtp", "SpecDecoding"),
        ([5], [3], [1], False, "eagle3", "ChunkedPrefill"),
        ([5], [3], [1], False, None, "ChunkedPrefill"),
        ([0, 5], [3, 1], [3, 1], True, None, "ChunkedPrefill"),
        ([0, 5], [3, 1], [3, 1], False, None, "PrefillCacheHit"),
        ([5], [3], [3], True, "mtp", "ChunkedPrefill"),
        ([5], [3], [3], False, "eagle3", "PrefillCacheHit"),
    ],
)
def test_ascend_phase_rules(computed, scheduled, valid, chunked, method, expected):
    helpers = _ascend_phase_helpers()
    actual = helpers.classify_ascend_attention_state(
        computed, scheduled, valid, chunked, method
    )
    assert actual is getattr(helpers.AscendAttentionState, expected)


@pytest.mark.parametrize(
    "method,use_mla,expected",
    [
        (None, False, "DecodeOnly"),
        ("eagle3", True, "DecodeOnly"),
        ("mtp", False, "ChunkedPrefill"),
        ("mtp", True, "SpecDecoding"),
    ],
)
def test_ascend_dummy_phase_rules(method, use_mla, expected):
    helpers = _ascend_phase_helpers()
    actual = helpers.get_ascend_dummy_attention_state(
        speculative_method=method, use_mla=use_mla
    )
    assert actual is getattr(helpers.AscendAttentionState, expected)


@pytest.mark.parametrize("method", [None, "mtp"])
def test_ascend_dummy_rejects_mixed_batch(method):
    helpers = _ascend_phase_helpers()
    with pytest.raises(NotImplementedError, match="create_mixed_batch"):
        helpers.get_ascend_dummy_attention_state(
            create_mixed_batch=True, speculative_method=method
        )


@pytest.mark.parametrize("with_runtime", [False, True])
def test_profile_finally_clears_decoder_once_and_restores_state(with_runtime):
    tree = ast.parse((ROOT / "worker/model_runner.py").read_text())
    cleanup = next(
        node.finalbody
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "set_cudagraph_capturing_enabled"
            for statement in node.finalbody
        )
    )
    calls = []
    wrapper_instance = NS(graph_pool="profile-pool")
    breakable_instance = NS(graph_pool="profile-breakable-pool")
    wrapper = NS(
        clear_all_graphs=lambda: calls.append("decoder"),
        _all_instances=[wrapper_instance],
    )
    breakable = NS(
        clear_all_graphs=lambda: calls.append("breakable"),
        _all_instances=[breakable_instance],
    )

    def clear_decoder_graphs():
        wrapper.clear_all_graphs()
        calls.append("tasks")

    dispatcher = NS(cudagraph_keys={"full": {1, 2}}, keys_initialized=True)
    runner = NS(
        graph_runtime=NS(clear_decoder_graphs=clear_decoder_graphs)
        if with_runtime
        else None,
        cudagraph_dispatcher=dispatcher,
        lora_config="lora-config",
        maybe_remove_all_loras=lambda config: calls.append(config),
        _cleanup_profiling_kv_cache=lambda: calls.append("kv-cache"),
    )
    counter = NS(num_cudagraph_captured=9)
    namespace = {
        "self": runner,
        "set_cudagraph_capturing_enabled": lambda enabled: calls.append(enabled),
        "GraphWrapper": wrapper,
        "CUDAGraphWrapper": NS(
            clear_all_graphs=lambda: calls.append("upstream"), _all_instances=[]
        ),
        "BreakableCUDAGraphWrapper": breakable,
        "encoder_cudagraph_manager": NS(clear=lambda: calls.append("encoder")),
        "original_pools": {
            id(wrapper_instance): "original-pool",
            id(breakable_instance): "original-breakable-pool",
        },
        "compilation_counter": counter,
        "saved_num_cudagraph_captured": 3,
    }
    exec(
        compile(ast.Module(body=cleanup, type_ignores=[]), "profile-finally", "exec"),
        namespace,
    )
    expected = [False, "upstream", "decoder"]
    if with_runtime:
        expected.append("tasks")
    assert calls == expected + ["breakable", "encoder", "lora-config", "kv-cache"]
    assert wrapper_instance.graph_pool == "original-pool"
    assert breakable_instance.graph_pool == "original-breakable-pool"
    assert dispatcher.cudagraph_keys == {"full": set()}
    assert not dispatcher.keys_initialized
    assert counter.num_cudagraph_captured == 3


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _graph_params(monkeypatch):
    name = "_fl_graph_params_contract"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "compilation/graph_params.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int = 4
    num_reqs: int = 4
    uniform: bool = True
    has_lora: bool = False
    num_active_loras: int = 0


@pytest.mark.parametrize(
    "first,second",
    [
        (_Descriptor(), _Descriptor(has_lora=True, num_active_loras=1)),
        (
            _Descriptor(has_lora=True, num_active_loras=1),
            _Descriptor(has_lora=True, num_active_loras=2),
        ),
        (_Descriptor(), _Descriptor(num_reqs=2)),
        (_Descriptor(), _Descriptor(uniform=False)),
    ],
)
def test_capture_registry_isolates_complete_descriptors(monkeypatch, first, second):
    registry = _graph_params(monkeypatch)
    params = registry.prepare_graph_params(first)
    task, handle, event, workspace = object(), object(), object(), object()
    params.attn_params[first].append(task)
    params.handles[first].append(handle)
    params.events[first].append(event)
    params.workspaces[first] = workspace

    registry.prepare_graph_params(second)
    params.require_capture(first)
    params.require_capture(second)
    assert params.attn_params[first] == [task]
    assert params.handles[first] == [handle]
    assert params.events[first] == [event]
    assert params.workspaces[first] is workspace
    assert params.attn_params[second] == []
    assert params.handles[second] == []
    assert params.events[second] == []
    assert params.workspaces[second] is None


def test_duplicate_descriptor_preserves_captured_state(monkeypatch):
    registry = _graph_params(monkeypatch)
    descriptor = _Descriptor()
    params = registry.prepare_graph_params(descriptor)
    task = object()
    params.attn_params[descriptor].append(task)
    with pytest.raises(RuntimeError, match="already contain captured state"):
        registry.prepare_graph_params(replace(descriptor))
    assert params.attn_params[descriptor] == [task]


def test_workspace_without_allocation_skips_conversion(monkeypatch):
    registry = _graph_params(monkeypatch)
    descriptor = _Descriptor()
    registry.prepare_graph_params(descriptor)

    def unexpected_conversion(_workspace):
        raise AssertionError("No workspace was allocated")

    registry.weak_ref_workspace(descriptor, unexpected_conversion)
    assert registry.get_graph_params().workspaces[descriptor] is None


def test_workspace_uses_platform_weak_ref_result(monkeypatch):
    registry = _graph_params(monkeypatch)
    descriptor = _Descriptor()
    params = registry.prepare_graph_params(descriptor)
    workspace, weak_workspace = object(), object()
    params.workspaces[descriptor] = workspace
    calls = []

    def weak_ref(tensor):
        calls.append(tensor)
        return weak_workspace

    registry.weak_ref_workspace(descriptor, weak_ref)
    assert calls == [workspace]
    assert params.workspaces[descriptor] is weak_workspace


def test_workspace_weak_ref_error_is_reported(monkeypatch):
    registry = _graph_params(monkeypatch)
    descriptor = _Descriptor()
    params = registry.prepare_graph_params(descriptor)
    workspace = object()
    params.workspaces[descriptor] = workspace

    def fail(_tensor):
        raise RuntimeError("platform weak-ref unavailable")

    with pytest.raises(RuntimeError, match="platform weak-ref unavailable"):
        registry.weak_ref_workspace(descriptor, fail)
    assert params.workspaces[descriptor] is workspace


def test_workspace_weak_ref_requires_prepared_capture(monkeypatch):
    registry = _graph_params(monkeypatch)
    with pytest.raises(RuntimeError, match="not prepared"):
        registry.weak_ref_workspace(_Descriptor(), lambda x: x)
    assert registry.get_graph_params().workspaces == {}


def test_graph_params_has_no_back_import_to_fl_graph():
    tree = ast.parse((ROOT / "compilation/graph_params.py").read_text())
    modules = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(name and name.startswith("vllm_fl") for name in modules)
    assert "vllm.utils.torch_utils" not in modules


def _runtime(monkeypatch, calls):
    stream = NS(synchronize=lambda: calls.append("synchronize"))
    platform = NS(
        device_type="npu",
        torch_device_fn=NS(
            current_stream=lambda: stream, Stream=lambda: "update-stream"
        ),
    )
    _module(monkeypatch, "vllm.platforms", current_platform=platform)
    _module(monkeypatch, "vllm.distributed.parallel_state", GraphCaptureContext=NS)
    _module(
        monkeypatch,
        "vllm_fl.compilation.graph_params",
        prepare_graph_params=lambda n: calls.append(("prepare", n)),
        weak_ref_workspace=lambda d, fn: calls.append(("weak-ref", d, fn)),
        clear_graph_params=lambda: calls.append("clear"),
    )
    _module(monkeypatch, "vllm_fl.compilation.graph", weak_ref_tensors=lambda x: x)
    spec = importlib.util.spec_from_file_location(
        "_fl_graph_contract_runtime", ROOT / "compilation/graph_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, platform


def test_capture_replay_orders_updates_after_enqueue(monkeypatch):
    calls = []
    runtime, _ = _runtime(monkeypatch, calls)
    controller = runtime.GraphRuntimeController(vllm_config="config")
    controller.attention_impls = (
        NS(update_graph_params=lambda _s, _c, d, _v: calls.append(("update", d))),
    )
    context = NS(batch_descriptor=_Descriptor(), capturing=False)
    with controller.capture_scope(context):
        assert context.capturing
        calls.append("capture")
    assert not context.capturing
    with controller.replay_scope(context):
        calls.append("replay")
    assert calls == [
        ("prepare", context.batch_descriptor),
        "capture",
        (
            "weak-ref",
            context.batch_descriptor,
            sys.modules["vllm_fl.compilation.graph"].weak_ref_tensors,
        ),
        "synchronize",
        "replay",
        ("update", context.batch_descriptor),
    ]


def test_failed_capture_and_replay_restore_state(monkeypatch):
    def fail_in_scope(scope, message):
        with scope:
            raise RuntimeError(message)

    calls = []
    runtime, _ = _runtime(monkeypatch, calls)
    controller = runtime.GraphRuntimeController(vllm_config="config")
    context = NS(batch_descriptor=_Descriptor(), capturing=False)
    pytest.raises(
        RuntimeError, fail_in_scope, controller.capture_scope(context), "capture failed"
    ).match("capture failed")
    assert not context.capturing and not controller._active
    calls.clear()
    pytest.raises(
        RuntimeError, fail_in_scope, controller.replay_scope(context), "replay failed"
    ).match("replay failed")
    assert calls == ["synchronize"]  # Failed enqueue must not release update events.
    assert not controller._active


def test_selected_backends_are_bound_once(monkeypatch):
    runtime, _ = _runtime(monkeypatch, [])
    base = type("AttentionBackend", (), {"get_impl_cls": staticmethod(lambda: None)})
    _module(monkeypatch, "vllm.v1.attention.backend", AttentionBackend=base)
    impl = type("Impl", (), {"update_graph_params": staticmethod(lambda *a: None)})
    backend = type("Backend", (base,), {"get_impl_cls": staticmethod(lambda: impl)})
    controller = runtime.GraphRuntimeController(vllm_config="config")
    controller.bind_attention_groups(
        [[NS(backend=backend), NS(backend=backend)], [NS(backend=base)]]
    )
    assert controller.attention_impls == (impl,)


def test_graph_cleanup_clears_wrapper_before_task_registry(monkeypatch):
    calls = []
    runtime, _ = _runtime(monkeypatch, calls)
    wrapper = NS(clear_all_graphs=lambda: calls.append("clear-wrappers"))
    controller = runtime.GraphRuntimeController(
        vllm_config="config", full_graph_wrapper_type=wrapper
    )
    controller.clear_decoder_graphs()
    assert calls == ["clear-wrappers", "clear"]


def _configure(monkeypatch, *, breakable=False):
    # Extract the standalone helper, avoiding PlatformFL's device discovery and
    # unrelated operator registration at import time on another vendor's CI.
    graph_modes = NS(NONE="none", FULL_DECODE_ONLY="decode", PIECEWISE="piecewise")
    compile_modes = NS(NONE="none", VLLM_COMPILE="vllm")
    _module(
        monkeypatch,
        "vllm.config",
        CUDAGraphMode=graph_modes,
        CompilationMode=compile_modes,
    )
    _module(monkeypatch, "vllm.envs", VLLM_USE_BREAKABLE_CUDAGRAPH=breakable)
    source = ast.parse((ROOT / "platform.py").read_text())
    helper = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "_configure_ascend_compilation"
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "platform-helper", "exec"),
        namespace,
    )
    return namespace[helper.name]


def _config(enabled=None, *, eager=False):
    options = {} if enabled is None else {"enable_npugraph_ex": enabled}
    return NS(
        additional_config={"ascend_compilation_config": options},
        model_config=NS(enforce_eager=eager),
        parallel_config=NS(use_ubatching=False),
        compilation_config=NS(mode="vllm", cudagraph_mode="decode", pass_config=NS()),
    )


@pytest.mark.parametrize("eager", [False, True])
def test_removed_npugraph_feature_rejected_even_in_eager(monkeypatch, eager):
    with pytest.raises(NotImplementedError, match="enable_npugraph_ex=true"):
        _configure(monkeypatch)(_config(True, eager=eager))


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_npugraph_flag_requires_boolean(monkeypatch, value):
    with pytest.raises(TypeError, match="must be bool"):
        _configure(monkeypatch)(_config(value))


COMPILER_OPTIONS = (
    "enable_npugraph_ex",
    "enable_static_kernel",
    "fuse_norm_quant",
    "fuse_qknorm_rope",
    "fuse_allreduce_rms",
    "fuse_muls_add",
)


@pytest.mark.parametrize("name", COMPILER_OPTIONS)
@pytest.mark.parametrize("eager", [False, True])
def test_removed_compiler_options_reject_enable(monkeypatch, name, eager):
    config = _config(eager=eager)
    config.additional_config["ascend_compilation_config"] = {name: True}
    with pytest.raises(NotImplementedError, match=f"{name}=true"):
        _configure(monkeypatch)(config)


@pytest.mark.parametrize("name", COMPILER_OPTIONS)
@pytest.mark.parametrize("value", [None, "false", 0, 1, [], {}])
def test_compiler_options_require_boolean(monkeypatch, name, value):
    config = _config()
    config.additional_config["ascend_compilation_config"] = {name: value}
    with pytest.raises(
        TypeError, match=f"ascend_compilation_config.{name} must be bool"
    ):
        _configure(monkeypatch)(config)


@pytest.mark.parametrize("value", [None, False, [], "disabled", 0])
def test_compiler_config_requires_dict(monkeypatch, value):
    config = _config(eager=True)
    config.additional_config["ascend_compilation_config"] = value
    with pytest.raises(TypeError, match="ascend_compilation_config must be a dict"):
        _configure(monkeypatch)(config)


@pytest.mark.parametrize("value", [False, [], "", 0])
def test_additional_config_requires_dict(monkeypatch, value):
    config = _config(eager=True)
    config.additional_config = value
    with pytest.raises(TypeError, match="additional_config must be a dict"):
        _configure(monkeypatch)(config)


@pytest.mark.parametrize("value", [False, True, None])
def test_unknown_compiler_option_rejected(monkeypatch, value):
    config = _config(eager=True)
    config.additional_config["ascend_compilation_config"] = {"typo_option": value}
    with pytest.raises(ValueError, match="Unknown ascend_compilation_config option"):
        _configure(monkeypatch)(config)


def test_explicit_disabled_compiler_options_supported(monkeypatch):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    config = _config()
    config.additional_config["ascend_compilation_config"] = dict.fromkeys(
        COMPILER_OPTIONS, False
    )
    _configure(monkeypatch)(config)
    assert config.compilation_config.backend == "eager"


def test_breakable_graph_rejected(monkeypatch):
    config = _config()
    with pytest.raises(NotImplementedError, match="VLLM_USE_BREAKABLE_CUDAGRAPH=1"):
        _configure(monkeypatch, breakable=True)(config)
    assert not hasattr(config.compilation_config, "backend")


@pytest.mark.parametrize("eager,mode", [(True, "decode"), (False, "none")])
def test_graph_disabled_allows_breakable_switch(monkeypatch, eager, mode):
    config = _config(eager=eager)
    config.compilation_config.cudagraph_mode = mode
    _configure(monkeypatch, breakable=True)(config)
    assert config.compilation_config.mode == "none"
    assert config.compilation_config.cudagraph_mode == "none"


@pytest.mark.parametrize("enabled", [None, False])
def test_full_decode_uses_eager_fx(monkeypatch, enabled):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    config = _config(enabled)
    _configure(monkeypatch)(config)
    compilation = config.compilation_config
    assert compilation.mode == "vllm" and compilation.backend == "eager"
    assert not compilation.use_inductor and compilation.splitting_ops == []
    assert compilation.cudagraph_num_of_warmups == 1
    assert not any(vars(compilation.pass_config).values())


def test_runtime_has_no_accelerator_import_side_effects():
    tree = ast.parse((ROOT / "compilation/graph_runtime.py").read_text())
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module for n in imports if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
    assert not any(
        name.startswith(("torch_npu", "npugraph_ex", "vllm_fl.attention"))
        for name in names
    )


def test_rotary_vendor_initialization_calls_registration_without_kernel(monkeypatch):
    calls = []
    _module(
        monkeypatch,
        "torch_npu.op_plugin.atb._atb_ops",
        _register_atb_extensions=lambda: calls.append("register"),
    )
    tree = ast.parse(
        (ROOT / "dispatch/backends/vendor/ascend/impl/rotary.py").read_text()
    )
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "initialize_rotary_embedding_ascend"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "rotary-init", "exec"),
        namespace,
    )
    namespace[helper.name]()
    assert calls == ["register"]
