# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def grid_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/gcu/impl/fla_grid.py"
    )
    spec = importlib.util.spec_from_file_location("gcu_fla_grid_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("programs", [1, 2, 255, 256, 65535, 65536, 65535 * 255])
def test_bounded_grid(grid_module, programs):
    x, y = grid_module.gcu_flat_grid(programs)
    assert 1 <= x <= 65535
    assert 1 <= y <= 255
    assert x * y >= programs
    assert x * (y - 1) < programs
    # The kernel's flattening visits every logical program exactly once.
    if programs <= 65536:
        visited = [
            a + b * x for b in range(y) for a in range(x) if a + b * x < programs
        ]
        assert visited == list(range(programs))


@pytest.mark.parametrize("programs", [0, -1, 65535 * 255 + 1])
def test_invalid_grid(grid_module, programs):
    with pytest.raises(ValueError):
        grid_module.gcu_flat_grid(programs)


@pytest.mark.parametrize("model_loaded", [False, True])
def test_install_updates_all_bindings(grid_module, monkeypatch, model_loaded):
    import sys
    from types import ModuleType

    def install_module(name):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            monkeypatch.setattr(sys.modules[parent], attr, module, raising=False)
        return module

    names = [
        "vllm",
        "vllm.third_party",
        "vllm.third_party.flash_linear_attention",
        "vllm.third_party.flash_linear_attention.ops",
        "vllm.third_party.flash_linear_attention.ops.chunk",
        "vllm.third_party.flash_linear_attention.ops.chunk_o",
        "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating",
        "fla_test",
        "fla_test.chunk_o_gcu",
        "fla_test.fused_sigmoid_gating_gcu",
    ]
    modules = {name: install_module(name) for name in names}

    def chunk():
        pass

    def gating():
        pass

    modules["fla_test.chunk_o_gcu"].chunk_fwd_o = chunk
    modules[
        "fla_test.fused_sigmoid_gating_gcu"
    ].fused_sigmoid_gating_delta_rule_update = gating
    grid_module.__package__ = "fla_test"
    model_name = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    model = ModuleType(model_name)
    if model_loaded:
        monkeypatch.setitem(sys.modules, model_name, model)
    else:
        monkeypatch.delitem(sys.modules, model_name, raising=False)
    for _ in range(2):
        grid_module.apply_fla_grid_gcu_patch()
    ops = "vllm.third_party.flash_linear_attention.ops"
    assert modules[ops + ".chunk"].chunk_fwd_o is chunk
    assert modules[ops + ".chunk_o"].chunk_fwd_o is chunk
    assert modules[ops].fused_sigmoid_gating_delta_rule_update is gating
    assert (
        modules[ops + ".fused_sigmoid_gating"].fused_sigmoid_gating_delta_rule_update
        is gating
    )
    if model_loaded:
        assert model.fused_sigmoid_gating_delta_rule_update is gating
    else:
        assert model_name not in sys.modules
