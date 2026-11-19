# Copyright (c) 2026 BAAI. All rights reserved.

"""Capability gate for the TXDA compact-grouped MoE fast path."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.ops.fused_moe import fused_moe_utils

_TX81_MODULE = "flag_gems.runtime.backend._tsingmicro.tx81.fused.fused_moe"


def _quant(**overrides):
    base = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _gate(**overrides):
    kwargs = dict(
        activation="silu",
        quant_config=_quant(),
        per_channel_quant=False,
        apply_router_weight_on_input=False,
        expert_map=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
        w1_scale=None,
        w2_scale=None,
    )
    kwargs.update(overrides)
    return fused_moe_utils._txda_fused_experts_impl(**kwargs)


@pytest.fixture
def txda_platform(monkeypatch):
    platform = SimpleNamespace(device_type="txda")

    class _Platforms:
        current_platform = platform

    monkeypatch.setitem(sys.modules, "vllm.platforms", _Platforms)
    return platform


@pytest.fixture
def flag_gems(monkeypatch):
    """A FlagGems build that exposes the pinned TX81 module."""
    module = ModuleType("flag_gems")
    module.__version__ = "5.5.0.dev1+g1c93bb08c"
    monkeypatch.setitem(sys.modules, "flag_gems", module)

    backend_pkg = ModuleType(_TX81_MODULE)
    sentinel = object()
    backend_pkg.fused_experts_impl = sentinel
    monkeypatch.setitem(
        sys.modules, _TX81_MODULE, backend_pkg
    )
    return sentinel


def test_plain_bf16_silu_takes_the_fast_path(txda_platform, flag_gems):
    assert _gate() is flag_gems


def test_non_txda_platform_never_takes_it(monkeypatch, flag_gems):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(device_type="cuda")),
    )
    assert _gate() is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"activation": "gelu"},
        {"apply_router_weight_on_input": True},
        {"expert_map": torch.zeros(2)},
        {"quant_config": _quant(use_fp8_w8a8=True)},
        {"quant_config": _quant(use_int8_w8a8=True)},
        {"quant_config": _quant(use_int4_w4a16=True)},
        {"block_shape": [128, 128]},
        {"a1_scale": torch.ones(1)},
        {"per_channel_quant": True},
        {"w1_bias": torch.ones(1)},
        {"w2_bias": torch.ones(1)},
        {"w1_scale": torch.ones(1)},
        {"quant_config": _quant(use_int8_w8a16=True)},
    ],
)
def test_unsupported_combinations_are_rejected(txda_platform, flag_gems, overrides):
    """Every case must be decided *before* running, never after."""
    assert _gate(**overrides) is None


def test_int8_w8a16_with_scales_is_supported(txda_platform, flag_gems):
    impl = _gate(
        quant_config=_quant(use_int8_w8a16=True),
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )
    assert impl is flag_gems


def test_old_flaggems_is_refused(txda_platform, monkeypatch):
    module = ModuleType("flag_gems")
    module.__version__ = "5.4.0"
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    assert _gate() is None


def test_missing_tx81_module_is_refused(txda_platform, monkeypatch):
    module = ModuleType("flag_gems")
    module.__version__ = "5.5.0.dev0"
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    # No entry in sys.modules and no real package -> ImportError.
    monkeypatch.setitem(sys.modules, "flag_gems.runtime.backend._tsingmicro.tx81", None)
    monkeypatch.setitem(sys.modules, "flag_gems.runtime.backend._tsingmicro", None)
    monkeypatch.delitem(
        sys.modules, _TX81_MODULE, raising=False
    )
    assert _gate() is None
