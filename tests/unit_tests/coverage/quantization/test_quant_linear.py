# Copyright (c) 2026 BAAI. All rights reserved.

"""Platform-resolution tests for vllm_fl.quantization.quant_linear.

``quant_linear`` imports ``vllm.platforms`` at module level, so each scenario
below installs a stub vllm platform graph and reloads the module: the
runtime platform object is bound at import time, which is exactly the
behaviour under test.  Only the external vllm boundary is stubbed;
``_resolve_source_platform`` and ``add_oot_quant_kernel`` run unmodified.

Existing tests (tests/unit_tests/quantization/test_w8a8_linear.py) already
cover the MXFP8 inheritance branch of ``add_oot_quant_kernel`` and the
kernel registration helpers; this module pins the platform-resolution
matrix and the remaining registry copies without duplicating those.
"""

import importlib
import sys
import types

import pytest

import vllm_fl.utils as utils  # noqa: F401  (ensures vllm_fl is importable)


def _make_module(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _build_platform_enum():
    """A PlatformEnum stub with hashable members, like the real enum."""
    import enum as _enum

    return _enum.Enum("PlatformEnum", ["CUDA", "ROCM", "CPU", "OOT", "XPU"])


def _install_quant_vllm_graph(platform):
    """Install the vllm graph quant_linear needs, with the given platform."""
    enum = _build_platform_enum()
    vllm = _make_module("vllm")
    platforms = _make_module("vllm.platforms")
    platforms.PlatformEnum = enum
    platforms.current_platform = platform
    vllm.platforms = platforms
    me = _make_module("vllm.model_executor")
    kernels = _make_module("vllm.model_executor.kernels")
    linear_kernels = _make_module("vllm.model_executor.kernels.linear")
    linear_kernels._POSSIBLE_KERNELS = {}
    linear_kernels._POSSIBLE_INT8_KERNELS = {}
    linear_kernels._POSSIBLE_FP8_KERNELS = {}
    linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS = {}
    linear_kernels._POSSIBLE_MXFP8_KERNELS = {}
    linear_kernels.Int8ScaledMMLinearKernel = type("Int8ScaledMMLinearKernel", (), {})
    linear_kernels.Int8ScaledMMLinearLayerConfig = type(
        "Int8ScaledMMLinearLayerConfig", (), {}
    )
    me.kernels = kernels
    kernels.linear = linear_kernels
    vllm.model_executor = me
    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.platforms": platforms,
            "vllm.model_executor": me,
            "vllm.model_executor.kernels": kernels,
            "vllm.model_executor.kernels.linear": linear_kernels,
        }
    )
    return enum, linear_kernels


def _stub_w8a8_installers(calls):
    """Redirect the three w8a8 installers so their real deps never import.

    ``add_oot_quant_kernel`` imports the installers from the w8a8 submodules
    (``from .w8a8.linear import register_fl_w8a8_linear_kernel`` etc.), which
    in turn need vllm/compressed_tensors classes unavailable on this machine.
    Installing lightweight function stubs on those submodules — before their
    first import via a sys.modules preload — keeps the code under test
    running unmodified while avoiding the unavailable dependencies.
    """
    w_linear = _make_module("vllm_fl.quantization.w8a8.linear")
    w_moe = _make_module("vllm_fl.quantization.w8a8.moe")
    w_packed = _make_module("vllm_fl.quantization.w8a8.packed")

    def _record_linear(registry):
        calls.append(("linear", registry))
        return True

    def _record_packed():
        calls.append(("packed", None))
        return True

    def _record_moe():
        calls.append(("moe", None))
        return True

    w_linear.register_fl_w8a8_linear_kernel = _record_linear
    w_moe.install_fl_w8a8_moe_selector = _record_moe
    w_packed.install_packed_w8a8_scheme = _record_packed
    sys.modules.update(
        {
            "vllm_fl.quantization.w8a8.linear": w_linear,
            "vllm_fl.quantization.w8a8.moe": w_moe,
            "vllm_fl.quantization.w8a8.packed": w_packed,
        }
    )
    return (w_linear, w_moe, w_packed)


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Clear every ``VLLM_FL_*`` env var for the test (restored on teardown).

    The supplementary conftest's helper of the same name: the tests below
    pin OOT / FlagGems policy off, and a stray ``VLLM_FL_*`` variable set
    by an earlier test would change what the module under test reads.
    """
    import os

    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch.setenv


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, clean_fl_env):
    """Keep OOT / FlagGems env vars unset around each test."""
    for var in (
        "VLLM_FL_OOT_ENABLED",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_PREFER",
        "USE_FLAGGEMS",
        "VLLM_FL_OOT_WHITELIST",
        "VLLM_FL_OOT_BLACKLIST",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    yield


def _reload_quant_linear(platform):
    """Install the graph and reload quant_linear; return (module, enum, stubs)."""
    enum, linear_kernels = _install_quant_vllm_graph(platform)
    import vllm_fl.quantization.quant_linear as quant_linear

    reloaded = importlib.reload(quant_linear)
    return reloaded, enum, linear_kernels


class TestResolveSourcePlatform:
    """_resolve_source_platform platform-check priority order."""

    @pytest.mark.parametrize(
        ("is_cuda_alike", "is_rocm", "is_cpu", "expected"),
        [
            (True, False, False, "cuda"),
            (True, True, False, "cuda"),  # cuda_alike wins over rocm
            (False, True, False, "rocm"),
            (False, False, True, "cpu"),
            (False, False, False, "cuda"),  # fallback
        ],
    )
    def test_platform_matrix(self, is_cuda_alike, is_rocm, is_cpu, expected):
        """Each runtime platform check maps to the documented source list."""
        platform = types.SimpleNamespace(
            is_cuda_alike=lambda: is_cuda_alike,
            is_rocm=lambda: is_rocm,
            is_cpu=lambda: is_cpu,
        )
        quant_linear, enum, _ = _reload_quant_linear(platform)
        assert quant_linear._resolve_source_platform() is enum[expected.upper()]

    def test_check_order_cuda_before_rocm(self):
        """is_cuda_alike is consulted before is_rocm."""
        calls = []

        platform = types.SimpleNamespace(
            is_cuda_alike=lambda: calls.append("cuda_alike") or True,
            is_rocm=lambda: calls.append("rocm") or True,
            is_cpu=lambda: calls.append("cpu") or False,
        )
        quant_linear, _, _ = _reload_quant_linear(platform)
        assert quant_linear._resolve_source_platform()
        assert calls == ["cuda_alike"]


class TestAddOotQuantKernel:
    """add_oot_quant_kernel copies every registry to the OOT key."""

    def test_all_five_registries_populated(self):
        """OOT lists are copies of the CUDA source lists (not the same object)."""
        kernel_a, kernel_b = object(), object()
        int8_a = object()
        fp8_a = object()
        fp8_block_a = object()
        mxfp8_a = object()
        platform = types.SimpleNamespace(
            is_cuda_alike=lambda: True,
            is_rocm=lambda: False,
            is_cpu=lambda: False,
        )
        quant_linear, enum, stubs = _reload_quant_linear(platform)
        stubs._POSSIBLE_KERNELS[enum.CUDA] = [kernel_a, kernel_b]
        stubs._POSSIBLE_INT8_KERNELS[enum.CUDA] = [int8_a]
        stubs._POSSIBLE_FP8_KERNELS[enum.CUDA] = [fp8_a]
        stubs._POSSIBLE_FP8_BLOCK_KERNELS[enum.CUDA] = [fp8_block_a]
        stubs._POSSIBLE_MXFP8_KERNELS[enum.CUDA] = [mxfp8_a]

        calls = []
        _stub_w8a8_installers(calls)
        quant_linear.add_oot_quant_kernel()

        assert stubs._POSSIBLE_KERNELS[enum.OOT] == [kernel_a, kernel_b]
        assert (
            stubs._POSSIBLE_KERNELS[enum.OOT] is not stubs._POSSIBLE_KERNELS[enum.CUDA]
        )
        assert stubs._POSSIBLE_INT8_KERNELS[enum.OOT] == [int8_a]
        assert stubs._POSSIBLE_FP8_KERNELS[enum.OOT] == [fp8_a]
        assert stubs._POSSIBLE_FP8_BLOCK_KERNELS[enum.OOT] == [fp8_block_a]
        assert stubs._POSSIBLE_MXFP8_KERNELS[enum.OOT] == [mxfp8_a]
        # The installers ran, in the documented order, with the INT8 registry.
        assert [name for name, _ in calls] == ["linear", "packed", "moe"]
        assert calls[0][1] is stubs._POSSIBLE_INT8_KERNELS

    def test_idempotent_second_call_keeps_existing_lists(self):
        """A second call does not overwrite or duplicate OOT entries."""
        platform = types.SimpleNamespace(
            is_cuda_alike=lambda: True,
            is_rocm=lambda: False,
            is_cpu=lambda: False,
        )
        quant_linear, enum, stubs = _reload_quant_linear(platform)
        native = object()
        extra = object()
        stubs._POSSIBLE_KERNELS[enum.CUDA] = [native]
        stubs._POSSIBLE_KERNELS[enum.OOT] = [extra]

        _stub_w8a8_installers([])
        quant_linear.add_oot_quant_kernel()

        assert stubs._POSSIBLE_KERNELS[enum.OOT] == [extra]

    def test_missing_source_list_gives_empty_oot_list(self):
        """A source platform with no registered list yields an empty OOT list."""
        platform = types.SimpleNamespace(
            is_cuda_alike=lambda: False,
            is_rocm=lambda: False,
            is_cpu=lambda: True,
        )
        quant_linear, enum, stubs = _reload_quant_linear(platform)
        # CPU has no _POSSIBLE_KERNELS entry at all.
        assert enum.CPU not in stubs._POSSIBLE_KERNELS

        _stub_w8a8_installers([])
        quant_linear.add_oot_quant_kernel()

        assert stubs._POSSIBLE_KERNELS[enum.OOT] == []
        assert stubs._POSSIBLE_INT8_KERNELS[enum.OOT] == []
