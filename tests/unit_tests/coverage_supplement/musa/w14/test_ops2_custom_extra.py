# Copyright (c) 2026 BAAI. All rights reserved.
"""Extra coverage for vllm_fl/ops/custom_ops.py registration branches."""

import sys
from types import ModuleType, SimpleNamespace

from vllm_fl.ops import custom_ops


def _patch_utils(monkeypatch, **values):
    import vllm_fl.utils as fl_utils

    defaults = {
        "is_oot_enabled": lambda: True,
        "get_oot_blacklist": lambda: [],
        "get_oot_whitelist": lambda: None,
        "use_flaggems_op": lambda op_name: True,
    }
    defaults.update(values)
    for name, value in defaults.items():
        monkeypatch.setattr(fl_utils, name, value)


def _patch_platform(monkeypatch, device_type="musa", vendor_name="mthreads"):
    import vllm.platforms as platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type=device_type, vendor_name=vendor_name),
    )


def _patch_platform_name(monkeypatch, name="musa"):
    import vllm_fl.dispatch.config.utils as cfg_utils

    monkeypatch.setattr(cfg_utils, "get_platform_name", lambda: name)


def _setup(monkeypatch, oot_ops, **utils):
    monkeypatch.setattr(custom_ops, "OOT_OPS", oot_ops)
    _patch_utils(monkeypatch, **utils)
    _patch_platform(monkeypatch)
    _patch_platform_name(monkeypatch)

    custom_calls = []
    pluggable_calls = []

    class DummyCustom:
        pass

    class DummyPluggable:
        pass

    DummyCustom.register_oot = staticmethod(
        lambda **kwargs: custom_calls.append(kwargs)
    )
    DummyPluggable.register_oot = staticmethod(
        lambda **kwargs: pluggable_calls.append(kwargs)
    )
    monkeypatch.setattr(custom_ops, "CustomOp", DummyCustom)
    monkeypatch.setattr(custom_ops, "PluggableLayer", DummyPluggable)

    oracle_calls = []
    factory_calls = []
    monkeypatch.setattr(
        custom_ops,
        "_patch_unquantized_moe_oracle",
        lambda: oracle_calls.append(True),
    )
    monkeypatch.setattr(
        custom_ops,
        "_patch_fused_moe_factory",
        lambda: factory_calls.append(True),
    )
    return custom_calls, pluggable_calls, oracle_calls, factory_calls


def test_unquantized_method_registers_when_flaggems_enabled(monkeypatch):
    class FakeCustom:
        pass

    calls, _, oracle_calls, factory_calls = _setup(
        monkeypatch,
        {"unquantized_fused_moe_method": (FakeCustom, "UFM")},
        use_flaggems_op=lambda op_name: True,
    )
    custom_ops.register_oot_ops()
    assert [call["name"] for call in calls] == ["UFM"]
    assert oracle_calls == [True]
    assert factory_calls == [True]


def test_patch_fused_moe_factory_early_returns_when_already_patched(monkeypatch):
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer_module

    monkeypatch.setattr(
        layer_module, "FusedMoE", custom_ops.FusedMoEFL, raising=False
    )
    monkeypatch.setattr(
        layer_module, "FusedMoEFactory", object(), raising=False
    )
    package_sentinel = object()
    package_factory_sentinel = object()
    monkeypatch.setattr(
        package, "FusedMoE", package_sentinel, raising=False
    )
    monkeypatch.setattr(
        package, "FusedMoEFactory", package_factory_sentinel, raising=False
    )

    custom_ops._patch_fused_moe_factory()

    assert package.FusedMoE is package_sentinel
    assert package.FusedMoEFactory is package_factory_sentinel


def test_register_oot_ops_blacklist_all_skips_factory(monkeypatch):
    class FakeCustom:
        pass

    calls, _, oracle_calls, factory_calls = _setup(
        monkeypatch,
        {"a": (FakeCustom, "A")},
        get_oot_blacklist=lambda: ["a", "fused_moe"],
    )
    custom_ops.register_oot_ops()
    assert [call["name"] for call in calls] == []
    assert oracle_calls == [True]
    assert factory_calls == []


def test_register_oot_ops_unknown_whitelist_entry_is_logged(monkeypatch, caplog):
    class FakeCustom:
        pass

    calls, _, _, _ = _setup(
        monkeypatch,
        {"a": (FakeCustom, "A")},
        get_oot_whitelist=lambda: ["missing", "a"],
    )
    custom_ops.register_oot_ops()
    assert [call["name"] for call in calls] == ["A"]
    assert "not found in OOT_OPS" in caplog.text
