# SPDX-License-Identifier: Apache-2.0
"""Graph contracts with runtime stubs; importing this test requires no NPU."""

import ast
import importlib.util
import os
import sys
from contextlib import contextmanager
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
        "BreakableCUDAGraphWrapper": breakable,
        "encoder_cudagraph_manager": NS(clear=lambda: calls.append("encoder")),
        "original_pools": {
            id(wrapper_instance): "original-pool",
            id(breakable_instance): "original-breakable-pool",
        },
        "compilation_counter": counter,
        "saved_num_cudagraph_captured": 3,
        "metadata_graph": NS(graph_pool="profile-metadata"),
        "original_metadata_pool": "original-metadata",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_clear_decoder_graphs", "_decoder_graph_wrappers")
    ]
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "graph-helpers", "exec"),
        namespace,
    )
    exec(
        compile(ast.Module(body=cleanup, type_ignores=[]), "profile-finally", "exec"),
        namespace,
    )
    expected = [False, "decoder"]
    if with_runtime:
        expected.append("tasks")
    assert calls == expected + ["breakable", "encoder", "lora-config", "kv-cache"]
    assert wrapper_instance.graph_pool == "original-pool"
    assert breakable_instance.graph_pool == "original-breakable-pool"
    assert dispatcher.cudagraph_keys == {"full": set()}
    assert not dispatcher.keys_initialized
    assert counter.num_cudagraph_captured == 3
    assert namespace["metadata_graph"].graph_pool == "original-metadata"


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


def _legacy_weak_ref_helper():
    tree = ast.parse((ROOT / "compilation/graph.py").read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "weak_ref_tensors"
    )
    namespace = {"Any": object}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "legacy-weakref", "exec"),
        namespace,
    )
    return namespace[helper.name]


def test_workspace_weak_ref_without_workspace_skips_upstream_import(monkeypatch):
    registry = _graph_params(monkeypatch)
    registry.prepare_graph_params(4)
    imports = []
    original_import = __import__

    def track_import(name, *args, **kwargs):
        imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", track_import)
    registry.weak_ref_workspace(4)
    assert registry.get_graph_params().workspaces[4] is None
    assert "vllm.utils.torch_utils" not in imports
    assert "vllm_fl.compilation.graph" not in imports


def test_workspace_weak_ref_saves_upstream_result(monkeypatch):
    registry = _graph_params(monkeypatch)
    params = registry.prepare_graph_params(4)
    workspace, weak_workspace = object(), object()
    params.workspaces[4] = workspace
    calls = []

    def weak_ref_tensors(tensor):
        calls.append(tensor)
        return weak_workspace

    _module(monkeypatch, "vllm.utils.torch_utils", weak_ref_tensors=weak_ref_tensors)
    registry.weak_ref_workspace(4)
    assert calls == [workspace]
    assert params.workspaces[4] is weak_workspace
    assert _legacy_weak_ref_helper()(workspace) is weak_workspace


@pytest.mark.parametrize("failure", ["import", "missing-helper", "runtime", "type"])
def test_workspace_weak_ref_failure_preserves_original(monkeypatch, failure):
    registry = _graph_params(monkeypatch)
    params = registry.prepare_graph_params(4)
    workspace = object()
    params.workspaces[4] = workspace
    calls = []

    def fail(tensor):
        calls.append(tensor)
        exception = RuntimeError if failure == "runtime" else TypeError
        raise exception("upstream weak-ref failed")

    if failure == "import":
        monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", None)
    elif failure == "missing-helper":
        _module(monkeypatch, "vllm.utils.torch_utils")
    else:
        _module(monkeypatch, "vllm.utils.torch_utils", weak_ref_tensors=fail)
    registry.weak_ref_workspace(4)
    assert params.workspaces[4] is workspace
    assert calls == ([workspace] if failure in ("runtime", "type") else [])
    assert _legacy_weak_ref_helper()(workspace) is workspace


def test_workspace_weak_ref_requires_prepared_shape(monkeypatch):
    registry = _graph_params(monkeypatch)
    with pytest.raises(RuntimeError, match="not prepared for capture size 4"):
        registry.weak_ref_workspace(4)
    assert registry.get_graph_params().workspaces == {}


def test_graph_params_has_no_back_import_to_fl_graph():
    tree = ast.parse((ROOT / "compilation/graph_params.py").read_text())
    modules = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    modules += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name and name.startswith("vllm_fl") for name in modules)
    assert "vllm.utils.torch_utils" in modules


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
        weak_ref_workspace=lambda n: calls.append(("weak-ref", n)),
        clear_graph_params=lambda: calls.append("clear"),
    )
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
        NS(update_graph_params=lambda *a: calls.append("update")),
    )
    context = NS(batch_descriptor=NS(num_tokens=4), capturing=False)
    with controller.capture_scope(context):
        assert context.capturing
        calls.append("capture")
    assert not context.capturing
    with controller.replay_scope(context):
        calls.append("replay")
    assert calls == [
        ("prepare", 4),
        "capture",
        ("weak-ref", 4),
        "synchronize",
        "replay",
        "update",
    ]


def test_failed_capture_and_replay_restore_state(monkeypatch):
    def fail_in_scope(scope, message):
        with scope:
            raise RuntimeError(message)

    calls = []
    runtime, _ = _runtime(monkeypatch, calls)
    controller = runtime.GraphRuntimeController(vllm_config="config")
    context = NS(batch_descriptor=NS(num_tokens=4), capturing=False)
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


def test_non_npu_capture_uses_existing_policy(monkeypatch):
    calls = []
    runtime, platform = _runtime(monkeypatch, calls)
    platform.device_type = "gcu"

    @contextmanager
    def existing_capture(*, device):
        calls.append(device)
        yield "existing-context"

    with runtime.get_graph_capture(existing_capture)("device") as context:
        assert context == "existing-context"
    assert calls == ["device"]


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


def _rotary_class(monkeypatch, device_type, calls):
    from typing import Optional

    import torch

    class Base:
        def __init__(self, head_size, rotary_dim, max_positions, base, neox, dtype):
            self.head_size = head_size
            self.rotary_dim = rotary_dim
            self.is_neox_style = neox
            # Deliberately different from query dtype: FL converts device only.
            self.cos_sin_cache = torch.ones(max_positions, rotary_dim, dtype=dtype)
            self.cache_writes = 0

        def __setattr__(self, name, value):
            if name == "cos_sin_cache" and hasattr(self, "cache_writes"):
                self.cache_writes += 1
            object.__setattr__(self, name, value)

    def rotary(obj, query, key, cos, sin, positions, interleaved, inplace):
        calls.append(("dispatch", cos.dtype, interleaved, inplace))
        return query, key

    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary",
        initialize_rotary_embedding_ascend=lambda: calls.append("initialize"),
    )
    tree = ast.parse((ROOT / "ops/rotary_embedding.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "RotaryEmbedding": Base,
        "current_platform": NS(device_type=device_type),
        "_rotary_embedding": rotary,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "rotary", "exec"), namespace)
    return namespace[cls.name]


@pytest.mark.parametrize("device_type", ["npu", "cuda", "musa"])
@pytest.mark.parametrize("compiling", [False, True])
def test_rotary_cache_write_and_initialization_contract(
    monkeypatch, device_type, compiling
):
    import torch

    calls = []
    cls = _rotary_class(monkeypatch, device_type, calls)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    obj = cls(4, 4, 8, 10000.0, True, torch.float64)
    assert calls == (["initialize"] if device_type == "npu" else [])
    query, key = torch.ones(2, 4), torch.ones(2, 4)
    cache = obj.cos_sin_cache
    for _ in range(2):
        q, k = obj.forward_oot(torch.tensor([0, 1]), query, key)
        assert torch.equal(q, query) and torch.equal(k, key)
    assert obj.cache_writes == (0 if device_type == "npu" and compiling else 2)
    assert obj.cos_sin_cache is cache
    assert calls[-2:] == [("dispatch", torch.float64, False, True)] * 2
    assert calls.count("initialize") == (1 if device_type == "npu" else 0)


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


def test_npu_compiling_consumes_local_converted_cache(monkeypatch):
    import torch

    calls = []
    cls = _rotary_class(monkeypatch, "npu", calls)
    obj = cls(4, 4, 8, 10000.0, True, torch.float64)
    converted = torch.ones(8, 4, dtype=torch.float64)
    conversions = []

    def convert(device):
        conversions.append(device)
        return converted

    # The stored object deliberately has no chunk method: consuming the old
    # buffer instead of the converted local reference must fail this contract.
    original = NS(to=convert)
    obj.cos_sin_cache = original
    obj.cache_writes = 0
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    query, key = torch.ones(2, 4), torch.ones(2, 4)
    obj.forward_oot(torch.tensor([0, 1]), query, key)
    assert conversions == [torch.device("cpu")]
    assert obj.cos_sin_cache is original and obj.cache_writes == 0
    assert calls[-1] == ("dispatch", torch.float64, False, True)


def _memory_hook(monkeypatch, platform, calls):
    import builtins

    _module(monkeypatch, "vllm.platforms", current_platform=platform)
    patch_module = (
        "vllm_fl.dispatch.backends.vendor.ascend.patches.patch_torch_accelerator"
    )
    _module(
        monkeypatch, patch_module, patch_torch_accelerator=lambda: calls.append("patch")
    )
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == patch_module:
            calls.append("import-shim")
        return original_import(name, *args, **kwargs)

    tree = ast.parse((ROOT / "__init__.py").read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_ascend_torch_accelerator"
    )
    namespace = {"__builtins__": dict(vars(builtins), __import__=checked_import)}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "memory-hook", "exec"),
        namespace,
    )
    return namespace[helper.name]


@pytest.mark.parametrize(
    "platform",
    [
        NS(device_type="cpu"),
        NS(device_type="cuda"),
        NS(device_type="npu"),
        NS(device_type="npu", vendor_name="other"),
    ],
)
def test_memory_hook_skips_non_ascend_without_import(monkeypatch, platform):
    calls = []
    _memory_hook(monkeypatch, platform, calls)()
    assert calls == []


def test_memory_hook_imports_shim_for_ascend_npu(monkeypatch):
    calls = []
    _memory_hook(monkeypatch, NS(device_type="npu", vendor_name="ascend"), calls)()
    assert calls == ["import-shim", "patch"]


def _memory_shim(npu, accelerator):
    tree = ast.parse(
        (
            ROOT / "dispatch/backends/vendor/ascend/patches/patch_torch_accelerator.py"
        ).read_text()
    )
    nodes = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
    ]
    namespace = {"torch": NS(npu=npu, accelerator=accelerator)}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "memory-shim", "exec"),
        namespace,
    )
    return namespace


MEMORY_APIS = (
    "empty_cache",
    "memory_stats",
    "memory_reserved",
    "reset_peak_memory_stats",
    "mem_get_info",
)


def test_memory_shim_api_mapping_and_idempotence():
    calls = []

    def api(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return call

    npu = NS(**{name: api(name) for name in MEMORY_APIS})
    accelerator = NS()
    namespace = _memory_shim(npu, accelerator)
    patch = namespace["patch_torch_accelerator"]
    patch()
    initial = vars(accelerator).copy()
    patch()
    assert vars(accelerator) == initial
    for target, source in namespace["_NPU_MEMORY_API_MAP"].items():
        assert getattr(accelerator, target) is getattr(npu, source)
        assert getattr(accelerator, target)(3, tag="probe") == source
    assert len(calls) == 5


@pytest.mark.parametrize("missing", MEMORY_APIS)
def test_memory_shim_missing_api_is_atomic(missing):
    npu = NS(**{name: lambda: None for name in MEMORY_APIS if name != missing})
    accelerator = NS(empty_cache="original", memory_reserved="original")
    original = vars(accelerator).copy()
    with pytest.raises(RuntimeError, match="missing required memory APIs"):
        _memory_shim(npu, accelerator)["patch_torch_accelerator"]()
    assert vars(accelerator) == original


@pytest.mark.parametrize("missing", ["npu", "accelerator"])
def test_memory_shim_missing_runtime_rejected(missing):
    npu = NS(**{name: lambda: None for name in MEMORY_APIS})
    accelerator = NS()
    namespace = _memory_shim(
        None if missing == "npu" else npu,
        None if missing == "accelerator" else accelerator,
    )
    with pytest.raises(RuntimeError, match=f"requires torch.{missing}"):
        namespace["patch_torch_accelerator"]()


def test_memory_shim_detects_modified_installation():
    npu = NS(**{name: lambda: None for name in MEMORY_APIS})
    accelerator = NS()
    patch = _memory_shim(npu, accelerator)["patch_torch_accelerator"]
    patch()
    accelerator.empty_cache = lambda: None
    with pytest.raises(RuntimeError, match="modified after installation"):
        patch()
