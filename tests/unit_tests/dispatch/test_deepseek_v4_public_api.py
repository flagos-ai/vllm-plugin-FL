# SPDX-License-Identifier: Apache-2.0
"""Dispatch and public API contracts for the DeepSeek V4 adapter."""

import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.backends.flaggems.impl import deepseek_v4 as adapter
from vllm_fl.dispatch.registry import OpRegistry


def _with_public_signature(op_name, function):
    function.__signature__ = inspect.Signature(
        [
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            *[
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
                for name in adapter._EXPLICIT_PARAMETERS[op_name]
            ],
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
    )
    return function


@pytest.fixture(autouse=True)
def clear_resolved():
    adapter._RESOLVED_APIS.clear()
    yield
    adapter._RESOLVED_APIS.clear()


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_public_calls_registered_independently_of_flaggems(monkeypatch, enabled):
    import vllm.platforms

    monkeypatch.setenv("USE_FLAGGEMS", enabled)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", enabled)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="ascend")
    )
    calls = {
        name: Mock(
            return_value=("attention", None) if name == "sparse_attn_sharedkv" else name
        )
        for name in adapter._PUBLIC_APIS.values()
    }
    for op_name, public_name in adapter._PUBLIC_APIS.items():
        _with_public_signature(op_name, calls[public_name])
    monkeypatch.setattr(
        adapter.importlib, "import_module", lambda name: SimpleNamespace(**calls)
    )
    registry = OpRegistry()
    adapter.register_deepseek_v4_operators(registry)
    impls = {
        name: registry.get_implementation(name, "default.flaggems_vllm")
        for name in adapter._PUBLIC_APIS
    }
    assert all(impl.is_available() for impl in impls.values())

    rope = SimpleNamespace(shape=(32, 64))
    args = [object() for _ in range(11)] + [rope]
    result = impls["deepseek_v4_compressor"].fn(*args, compress_ratio=128)
    assert result == "compressor"
    positional, kw = calls["compressor"].call_args
    assert positional == (args[0], args[1], args[2], args[4], args[3], args[10])
    assert kw == dict(
        state_block_table=args[7],
        cmp_ratio=128,
        coff=1,
        rope_head_dim=64,
        norm_eps=1e-6,
        positions=args[5],
        slot_mapping=args[6],
        output_request_ids=args[8],
        output_positions=args[9],
        cos_sin_cache=rope,
    )

    sparse_args = [object() for _ in range(10)]
    assert (
        impls["deepseek_v4_sparse_attention"].fn(*sparse_args, 128, 4, 0.0)
        == "attention"
    )
    positional, kw = calls["sparse_attn_sharedkv"].call_args
    assert positional == (sparse_args[0],)
    assert kw["ori_kv"] is sparse_args[1] and kw["cmp_kv"] is sparse_args[2]
    assert (
        kw["ori_block_table"] is sparse_args[3]
        and kw["cmp_block_table"] is sparse_args[4]
    )
    assert kw["request_ids"] is sparse_args[5] and kw["positions"] is sparse_args[6]
    assert kw["ori_lengths"] is sparse_args[7] and kw["cmp_lengths"] is sparse_args[8]
    assert kw["cmp_sparse_indices"] is sparse_args[9]
    assert (
        kw["ori_win_left"] == 127
        and kw["cmp_ratio"] == 4
        and kw["softmax_scale"] == 0.0
    )

    index_args = [object() for _ in range(8)]
    assert (
        impls["deepseek_v4_indexer"].fn(
            *index_args, query_scale="q", key_scale_cache="k"
        )
        == "quant_lightning_indexer"
    )
    calls["quant_lightning_indexer"].assert_called_once_with(
        *index_args, compress_ratio=4, query_scale="q", key_scale_cache="k"
    )
    values = SimpleNamespace(ndim=2, shape=(3, 8))
    slots = SimpleNamespace(shape=(3,))
    cache = SimpleNamespace(ndim=3, shape=(2, 4, 8))
    assert impls["paged_cache_write"].fn(values, slots, cache) is None
    calls["paged_scatter_triton"].assert_called_once_with(cache, slots, values, 4)
    calls["paged_scatter_triton"].return_value = False
    with pytest.raises(NotImplementedError, match="Unsupported paged cache"):
        impls["paged_cache_write"].fn(values, slots, cache)


def test_non_ascend_does_not_import_library(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="cuda")
    )

    def forbidden(name):
        pytest.fail(f"unexpected import: {name}")

    monkeypatch.setattr(adapter.importlib, "import_module", forbidden)
    assert not adapter.is_available("deepseek_v4_compressor")


@pytest.mark.parametrize("api", [None, lambda: None, lambda *args, **kwargs: None])
def test_missing_or_incompatible_capability_is_unavailable(monkeypatch, api):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="ascend")
    )
    monkeypatch.setattr(
        adapter.importlib, "import_module", lambda name: SimpleNamespace(compressor=api)
    )
    assert not adapter.is_available("deepseek_v4_compressor")
    assert "deepseek_v4_compressor" not in adapter._RESOLVED_APIS


@pytest.fixture
def dispatch_manager(monkeypatch):
    import os

    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.policy import (
        SelectionPolicy,
        reset_global_policy,
        set_global_policy,
    )

    registry = OpRegistry()
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    reset_global_policy()
    set_global_policy(SelectionPolicy(strict=False))
    monkeypatch.setattr(dispatch, "_CACHED_OPS", [])
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    monkeypatch.setattr(dispatch, "is_dump_enabled", lambda: False)
    yield manager, registry
    reset_global_policy()


@pytest.mark.parametrize("mode", ["cached", "manager", "fast_path_disabled"])
@pytest.mark.parametrize("op_name", list(adapter._PUBLIC_APIS))
def test_public_execution_error_is_never_retried(
    monkeypatch, dispatch_manager, mode, op_name
):
    import vllm.platforms

    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    manager, registry = dispatch_manager
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="ascend")
    )
    error = RuntimeError("public execution failed")
    failed = _with_public_signature(op_name, Mock(side_effect=error))
    fallback = Mock(return_value="fallback")
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: SimpleNamespace(**{adapter._PUBLIC_APIS[op_name]: failed}),
    )
    adapter.register_deepseek_v4_operators(registry)
    registry.register_impl(
        OpImpl(op_name, "reference.test", BackendImplKind.REFERENCE, fallback)
    )
    args = {
        "deepseek_v4_compressor": [object()] * 11 + [SimpleNamespace(shape=(8, 2))],
        "deepseek_v4_sparse_attention": [object()] * 10 + [128],
        "deepseek_v4_indexer": [object()] * 8,
        "paged_cache_write": [
            SimpleNamespace(ndim=2, shape=(3, 8)),
            SimpleNamespace(shape=(3,)),
            SimpleNamespace(ndim=3, shape=(2, 4, 8)),
        ],
    }[op_name]
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", mode != "fast_path_disabled")
    invoke = manager.call if mode == "manager" else dispatch.CachedOp(op_name)
    with pytest.raises(RuntimeError) as caught:
        if mode == "manager":
            invoke(op_name, *args)
        else:
            invoke(*args)
    assert caught.value is error
    failed.assert_called_once()
    fallback.assert_not_called()


@pytest.mark.parametrize("mode", ["cached", "manager", "fast_path_disabled"])
def test_default_provider_retains_fallback(monkeypatch, dispatch_manager, mode):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    manager, registry = dispatch_manager
    failed = Mock(side_effect=RuntimeError("failure"))
    fallback = Mock(return_value="fallback")
    registry.register_impl(
        OpImpl("test_op", "default.test", BackendImplKind.DEFAULT, failed)
    )
    registry.register_impl(
        OpImpl("test_op", "reference.test", BackendImplKind.REFERENCE, fallback)
    )
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", mode != "fast_path_disabled")
    if mode == "manager":
        result = manager.call("test_op", 7)
    else:
        result = dispatch.CachedOp("test_op")(7)
    assert result == "fallback"
    failed.assert_called_once_with(7)
    fallback.assert_called_once_with(7)


def test_explicit_policy_excludes_public_provider(monkeypatch, dispatch_manager):
    import vllm.platforms

    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    manager, registry = dispatch_manager
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="ascend")
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            compressor=_with_public_signature("deepseek_v4_compressor", Mock())
        ),
    )
    adapter.register_deepseek_v4_operators(registry)
    policy = SelectionPolicy.from_dict(
        strict=True, per_op_order={"deepseek_v4_compressor": ["reference"]}
    )
    with policy_context(policy), pytest.raises(RuntimeError, match="No implementation"):
        manager.resolve("deepseek_v4_compressor")


@pytest.mark.parametrize("mode", ["cached", "manager", "fast_path_disabled"])
def test_paged_cache_write_error_preserves_state_and_exception(
    monkeypatch, dispatch_manager, mode
):
    import vllm.platforms

    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    manager, registry = dispatch_manager
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="ascend")
    )
    error = RuntimeError("cache write failed after mutation")
    state = []

    def write(*args):
        state.append("written")
        raise error

    failed = _with_public_signature("paged_cache_write", Mock(side_effect=write))
    fallback = Mock(side_effect=lambda *args: state.append("retried"))
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: SimpleNamespace(paged_scatter_triton=failed),
    )
    adapter.register_deepseek_v4_operators(registry)
    registry.register_impl(
        OpImpl(
            "paged_cache_write", "reference.test", BackendImplKind.REFERENCE, fallback
        )
    )
    args = (
        SimpleNamespace(ndim=2, shape=(1, 8)),
        SimpleNamespace(shape=(1,)),
        SimpleNamespace(ndim=3, shape=(2, 4, 8)),
    )
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", mode != "fast_path_disabled")
    with pytest.raises(RuntimeError) as caught:
        if mode == "manager":
            manager.call("paged_cache_write", *args)
        else:
            dispatch.CachedOp("paged_cache_write")(*args)
    assert caught.value is error
    assert state == ["written"]
    failed.assert_called_once()
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "op_name",
    [
        "deepseek_v4_index_quant",
        "w8a8_prepare_weight",
        "w8a8_dynamic_linear",
        "w8a8_experts",
        "w8a8_prepare_experts",
    ],
)
def test_native_quantization_capability_preflight(monkeypatch, op_name):
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as native

    calls = {}
    for name, (count, keywords) in native._NATIVE_CALLS.items():
        fn = Mock()
        fn._schemas = {}
        fn.__signature__ = inspect.Signature(
            [
                *[
                    inspect.Parameter(f"arg{i}", inspect.Parameter.POSITIONAL_ONLY)
                    for i in range(count)
                ],
                *[
                    inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY)
                    for key in keywords
                ],
            ]
        )
        calls[name] = fn
    module = SimpleNamespace(**calls)
    monkeypatch.setattr(native.importlib, "import_module", lambda name: module)
    assert native._native_available(op_name, lambda: True)
    if native._NATIVE_REQUIREMENTS[op_name]:
        last = native._NATIVE_REQUIREMENTS[op_name][-1]
        delattr(module, last)
        assert not native._native_available(op_name, lambda: True)
    for fn in calls.values():
        fn.assert_not_called()
    monkeypatch.setattr(native.importlib, "import_module", Mock())
    assert not native._native_available(op_name, lambda: False)
    native.importlib.import_module.assert_not_called()


def test_native_quantization_packet_schema(monkeypatch):
    import torch

    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as native

    fn = Mock()
    fn._schemas = {
        "": torch._C.parse_schema(
            "test::dynamic(Tensor input, *, int? dst_type=None) -> (Tensor, Tensor)"
        )
    }
    monkeypatch.setattr(
        native.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            npu_dynamic_quant=fn,
        ),
    )
    assert native._native_available("deepseek_v4_index_quant", lambda: True)
    fn._schemas = {
        "": torch._C.parse_schema("test::dynamic(Tensor input) -> (Tensor, Tensor)")
    }
    assert not native._native_available("deepseek_v4_index_quant", lambda: True)
    fn.assert_not_called()


@pytest.mark.parametrize("mode", ["cached", "manager", "fast_path_disabled"])
@pytest.mark.parametrize(
    "op_name",
    [
        "deepseek_v4_index_quant",
        "w8a8_prepare_weight",
        "w8a8_dynamic_linear",
        "w8a8_experts",
        "w8a8_prepare_experts",
    ],
)
def test_native_quantization_execution_error_is_never_retried(
    monkeypatch,
    dispatch_manager,
    mode,
    op_name,
):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as native
    from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    manager, registry = dispatch_manager
    error = RuntimeError("native execution already launched")
    failed = Mock(side_effect=error)
    monkeypatch.setattr(AscendBackend, op_name, failed)
    monkeypatch.setattr(native, "_native_available", lambda *args: True)
    native.register_builtins(registry)
    impl = registry.get_implementation(op_name, "vendor.ascend")
    # Use the actual registration and availability hook; make selection device
    # independent without changing vendor policy or accessing an NPU.
    registry.clear()
    impl = replace(impl, vendor=None, kind=BackendImplKind.DEFAULT)
    registry.register_impl(impl)
    fallback = Mock(return_value="fallback")
    registry.register_impl(
        OpImpl(
            op_name,
            "reference.test",
            BackendImplKind.REFERENCE,
            fallback,
        )
    )
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", mode != "fast_path_disabled")
    with pytest.raises(RuntimeError) as caught:
        if mode == "manager":
            manager.call(op_name, 7)
        else:
            dispatch.CachedOp(op_name)(7)
    assert caught.value is error
    failed.assert_called_once_with(7)
    fallback.assert_not_called()
