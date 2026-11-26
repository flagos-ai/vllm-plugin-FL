# Copyright (c) 2026 BAAI. All rights reserved.

"""Capability gate for the TXDA compact-grouped MoE fast path.

The gate must reach the kernel through the *published* provider API
(``flaggems_vllm.fused_experts_impl``) only. These tests pin that contract:
they install a fake ``flaggems_vllm`` module and assert the gate never falls
back to FlagGems' private ``runtime.backend._tsingmicro`` tree.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.ops.fused_moe import fused_moe_utils


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


def _sentinel():
    def fused_experts_impl(*args, **kwargs):
        raise AssertionError("gate tests must not execute the kernel")

    return fused_experts_impl


def _install_provider(monkeypatch, version="0.3.0.dev1", impl=None):
    """Install a fake ``flaggems_vllm`` exposing the public fused entry point."""
    module = ModuleType("flaggems_vllm")
    module.__version__ = version
    sentinel = impl if impl is not None else _sentinel()
    module.fused_experts_impl = sentinel
    monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    return sentinel


def test_plain_bf16_silu_takes_the_fast_path(txda_platform, monkeypatch):
    impl = _install_provider(monkeypatch)
    assert _gate() is impl


def test_non_txda_platform_never_takes_it(monkeypatch):
    impl = _install_provider(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(device_type="cuda")),
    )
    assert _gate() is not impl
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
def test_unsupported_combinations_are_rejected(txda_platform, monkeypatch, overrides):
    """Every case must be decided *before* running, never after."""
    _install_provider(monkeypatch)
    assert _gate(**overrides) is None


def test_int8_w8a16_with_scales_is_supported(txda_platform, monkeypatch):
    impl = _install_provider(monkeypatch)
    got = _gate(
        quant_config=_quant(use_int8_w8a16=True),
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )
    assert got is impl


def test_capability_not_version_decides(txda_platform, monkeypatch):
    """A provider is accepted on the exported symbol alone, not its version.

    The pinned CI provider reports ``__version__ == "0.1.0"`` yet exports
    ``fused_experts_impl``; a version floor would wrongly disable the fast path
    there.
    """
    impl = _install_provider(monkeypatch, version="0.1.0")
    assert _gate() is impl


@pytest.mark.parametrize("version", ["0.1.0", "0.3.0.dev1", "0.3.0+abc123", "0.4.1"])
def test_any_version_exporting_the_symbol_is_accepted(
    txda_platform, monkeypatch, version
):
    impl = _install_provider(monkeypatch, version=version)
    assert _gate() is impl


def test_missing_public_symbol_is_refused(txda_platform, monkeypatch):
    module = ModuleType("flaggems_vllm")
    module.__version__ = "0.3.0"
    # No ``fused_experts_impl`` -> the gate must refuse rather than reach into a
    # private module path.
    monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    assert _gate() is None


def test_never_imports_the_private_flag_gems_tree(txda_platform, monkeypatch):
    """The provider must be reached only through its published API."""
    impl = _install_provider(monkeypatch)

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("flag_gems.runtime.backend._tsingmicro"):
                raise AssertionError(f"gate imported private module {name!r}")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])
    assert _gate() is impl
