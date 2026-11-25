# SPDX-License-Identifier: Apache-2.0
"""Dispatch contracts for DeepSeek V4 references and Ascend W8A8."""

import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.registry import OpRegistry


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


_REFERENCE_OPS = (
    "deepseek_v4_compressor",
    "deepseek_v4_sparse_attention",
    "deepseek_v4_indexer",
    "paged_cache_write",
)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_reference_dispatch_with_optional_flaggems(
    monkeypatch, dispatch_manager, enabled
):
    from vllm_fl.dispatch.backends.flaggems.register_ops import (
        register_builtins as register_flaggems,
    )
    from vllm_fl.dispatch.backends.reference.register_ops import register_builtins

    manager, registry = dispatch_manager
    monkeypatch.setenv("USE_FLAGGEMS", enabled)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "sigmoid")
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    register_flaggems(registry)
    register_builtins(registry)
    for op_name in _REFERENCE_OPS:
        assert manager.get_selected_impl_id(op_name) == "reference.torch"
        (impl,) = registry.get_implementations(op_name)
        assert not impl.allow_execution_fallback


def test_reference_paged_write_preserves_padding_and_strided_storage(dispatch_manager):
    import torch

    from vllm_fl.dispatch.backends.reference.register_ops import register_builtins

    manager, registry = dispatch_manager
    register_builtins(registry)
    storage = torch.full((2, 2, 7), -5, dtype=torch.float16)
    cache = storage[..., 1::2]
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    slots = torch.tensor([0, -1, 2**40, 3], dtype=torch.int64)
    manager.call("paged_cache_write", values, slots, cache)
    expected = torch.full_like(storage, -5)
    expected[0, 0, 1::2] = values[0]
    expected[1, 1, 1::2] = values[3]
    torch.testing.assert_close(storage, expected)

    with pytest.raises(ValueError, match="must not alias"):
        manager.call("paged_cache_write", cache[0], slots[:2], cache)
    torch.testing.assert_close(storage, expected)


@pytest.mark.parametrize("mode", ["cached", "manager", "fast_path_disabled"])
@pytest.mark.parametrize("op_name", _REFERENCE_OPS)
def test_reference_execution_error_is_never_retried(
    monkeypatch, dispatch_manager, mode, op_name
):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend
    from vllm_fl.dispatch.backends.reference.register_ops import register_builtins
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    manager, registry = dispatch_manager
    error = RuntimeError("reference execution failed after starting")
    state = []

    def execute(self, *args):
        state.append("started")
        raise error

    monkeypatch.setattr(ReferenceBackend, op_name, execute)
    register_builtins(registry)
    fallback = Mock()
    registry.register_impl(
        OpImpl(op_name, "reference.test", BackendImplKind.REFERENCE, fallback)
    )
    assert manager.get_selected_impl_id(op_name) == "reference.torch"
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", mode != "fast_path_disabled")
    with pytest.raises(RuntimeError) as caught:
        if mode == "manager":
            manager.call(op_name)
        else:
            dispatch.CachedOp(op_name)()
    assert caught.value is error
    assert state == ["started"]
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
