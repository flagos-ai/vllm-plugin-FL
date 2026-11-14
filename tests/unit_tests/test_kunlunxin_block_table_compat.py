# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Tests for the kunlunxin block_table compat shim.

The shim replaces the V2 runner's _compute_slot_mappings_kernel, which
FlagTree's XPU backend cannot lower because its padding program returns early
after emitting tensor work. These tests cover the patch mechanics; the
rewritten kernel's numerical equivalence is verified on hardware, since
compiling it requires the XPU Triton backend.
"""

import sys
import types

import pytest

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import block_table_compat


@pytest.fixture
def fake_block_table(monkeypatch):
    """Install a stand-in vllm.v1.worker.gpu.block_table module."""

    def original_kernel():  # pragma: no cover - never called
        raise AssertionError("the original kernel must not run")

    module = types.ModuleType("vllm.v1.worker.gpu.block_table")
    module._compute_slot_mappings_kernel = original_kernel
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.block_table", module)
    return module, original_kernel


@pytest.fixture
def stub_triton(monkeypatch):
    """Stub the Triton pieces so _build_kernel works without the XPU backend."""

    def jit(fn):
        return fn

    triton_module = types.ModuleType("vllm.triton_utils")
    triton_module.triton = types.SimpleNamespace(jit=jit)
    triton_module.tl = types.SimpleNamespace(constexpr=int, int32="int32")
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_module)

    buffer_utils = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffer_utils._load_ptr = lambda ptr, dtype: ptr
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.buffer_utils", buffer_utils)
    return triton_module


def test_exports_public_api():
    assert callable(block_table_compat.apply_block_table_compat)
    assert callable(block_table_compat._build_kernel)


def test_replaces_the_kernel(fake_block_table, stub_triton):
    module, original = fake_block_table

    block_table_compat.apply_block_table_compat()

    replaced = module._compute_slot_mappings_kernel
    assert replaced is not original
    assert getattr(replaced, block_table_compat._PATCH_FLAG) is True


def test_is_idempotent(fake_block_table, stub_triton):
    module, _ = fake_block_table

    block_table_compat.apply_block_table_compat()
    first = module._compute_slot_mappings_kernel
    block_table_compat.apply_block_table_compat()

    assert module._compute_slot_mappings_kernel is first


def test_does_not_override_a_foreign_replacement(fake_block_table, stub_triton):
    """A kernel someone else installed and flagged is left alone."""
    module, _ = fake_block_table

    def foreign_kernel():  # pragma: no cover - never called
        raise AssertionError("unreachable")

    setattr(foreign_kernel, block_table_compat._PATCH_FLAG, True)
    module._compute_slot_mappings_kernel = foreign_kernel

    block_table_compat.apply_block_table_compat()

    assert module._compute_slot_mappings_kernel is foreign_kernel


def test_tolerates_a_missing_kernel(fake_block_table, stub_triton):
    """A build whose module lacks the kernel is skipped, not crashed."""
    module, _ = fake_block_table
    del module._compute_slot_mappings_kernel

    block_table_compat.apply_block_table_compat()

    assert not hasattr(module, "_compute_slot_mappings_kernel")


def test_tolerates_a_build_without_the_v2_runner(monkeypatch):
    """Builds that cannot import the module are a no-op, not an error."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "vllm.v1.worker.gpu" or name.startswith(
            "vllm.v1.worker.gpu.block_table"
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu.block_table", raising=False)

    block_table_compat.apply_block_table_compat()


def test_rewritten_kernel_keeps_the_upstream_signature(fake_block_table, stub_triton):
    """The replacement must be call-compatible with upstream's launcher."""
    import inspect

    module, _ = fake_block_table
    block_table_compat.apply_block_table_compat()

    params = list(inspect.signature(module._compute_slot_mappings_kernel).parameters)
    assert params == [
        "max_num_tokens",
        "idx_mapping",
        "query_start_loc",
        "pos",
        "block_table_ptrs",
        "block_table_strides",
        "block_sizes",
        "slot_mapping_enabled",
        "dcp_sharded",
        "slot_mappings_ptr",
        "slot_mappings_stride",
        "cp_rank",
        "CP_SIZE",
        "CP_INTERLEAVE",
        "PAD_ID",
        "TRITON_BLOCK_SIZE",
    ]
