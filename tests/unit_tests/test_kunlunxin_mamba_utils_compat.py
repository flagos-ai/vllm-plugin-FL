# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Tests for the kunlunxin mamba_utils compat shim.

The shim replaces the V2 runner's aligned-state-indices kernel, which
FlagTree's XPU backend cannot lower because it gathers all Mamba groups
through a 3-D broadcast. These tests cover the patch mechanics; the
rewritten kernel's numerical equivalence is verified on hardware, since
compiling it requires the XPU Triton backend.
"""

import sys
import types

import pytest

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import mamba_utils_compat


@pytest.fixture
def fake_mamba_utils(monkeypatch):
    """Install a stand-in vllm.v1.worker.mamba_utils module."""

    def original_kernel():  # pragma: no cover - never called
        raise AssertionError("the original kernel must not run")

    module = types.ModuleType("vllm.v1.worker.mamba_utils")
    module.get_aligned_state_indices_multi_group_kernel = original_kernel
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.mamba_utils", module)
    return module, original_kernel


@pytest.fixture
def stub_triton(monkeypatch):
    """Stub vllm.triton_utils so _build_kernel works without the XPU backend."""

    def jit(*args, **kwargs):
        # triton.jit is used with keyword arguments here
        # (@triton.jit(do_not_specialize=[...])), so support both forms.
        if args and callable(args[0]):
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    triton = types.SimpleNamespace(jit=jit)
    tl = types.SimpleNamespace(constexpr=int, int64=int)
    module = types.ModuleType("vllm.triton_utils")
    module.triton = triton
    module.tl = tl
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", module)
    return module


def test_exports_public_api():
    assert callable(mamba_utils_compat.apply_mamba_utils_compat)
    assert callable(mamba_utils_compat._build_kernel)


def test_replaces_the_kernel(fake_mamba_utils, stub_triton):
    module, original = fake_mamba_utils

    mamba_utils_compat.apply_mamba_utils_compat()

    replaced = module.get_aligned_state_indices_multi_group_kernel
    assert replaced is not original
    assert getattr(replaced, mamba_utils_compat._PATCH_FLAG) is True


def test_is_idempotent(fake_mamba_utils, stub_triton):
    module, _ = fake_mamba_utils

    mamba_utils_compat.apply_mamba_utils_compat()
    first = module.get_aligned_state_indices_multi_group_kernel
    mamba_utils_compat.apply_mamba_utils_compat()

    assert module.get_aligned_state_indices_multi_group_kernel is first


def test_does_not_override_a_foreign_replacement(fake_mamba_utils, stub_triton):
    """A kernel someone else installed and flagged is left alone."""
    module, _ = fake_mamba_utils

    def foreign_kernel():  # pragma: no cover - never called
        raise AssertionError("unreachable")

    setattr(foreign_kernel, mamba_utils_compat._PATCH_FLAG, True)
    module.get_aligned_state_indices_multi_group_kernel = foreign_kernel

    mamba_utils_compat.apply_mamba_utils_compat()

    assert module.get_aligned_state_indices_multi_group_kernel is foreign_kernel


def test_tolerates_a_missing_kernel(fake_mamba_utils, stub_triton):
    """A build whose module lacks the kernel is skipped, not crashed."""
    module, _ = fake_mamba_utils
    del module.get_aligned_state_indices_multi_group_kernel

    mamba_utils_compat.apply_mamba_utils_compat()

    assert not hasattr(module, "get_aligned_state_indices_multi_group_kernel")


def test_tolerates_a_build_without_the_v2_mamba_path(monkeypatch):
    """Builds that cannot import the module are a no-op, not an error."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "vllm.v1.worker" or name.startswith("vllm.v1.worker.mamba_utils"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.mamba_utils", raising=False)

    mamba_utils_compat.apply_mamba_utils_compat()


def test_rewritten_kernel_keeps_the_upstream_signature(fake_mamba_utils, stub_triton):
    """The replacement must be call-compatible with upstream's launcher."""
    import inspect

    module, _ = fake_mamba_utils
    mamba_utils_compat.apply_mamba_utils_compat()

    params = list(
        inspect.signature(
            module.get_aligned_state_indices_multi_group_kernel
        ).parameters
    )
    assert params == [
        "block_table_ptrs_ptr",
        "seq_lens_ptr",
        "state_indices_ptr",
        "block_table_stride_req",
        "seq_lens_stride",
        "state_indices_stride_0",
        "state_indices_stride_1",
        "state_indices_stride_2",
        "num_requests",
        "CACHE_BLOCK_SIZE",
        "NUM_GROUPS",
        "BLOCK_GROUPS",
        "NUM_STATE_SLOTS",
        "BLOCK_STATE_SLOTS",
        "BLOCK_ROWS",
    ]
