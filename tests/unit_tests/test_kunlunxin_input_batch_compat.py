# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Tests for the kunlunxin input_batch compat shim.

The shim replaces the V2 runner's _combine_sampled_and_draft_tokens_kernel,
which FlagTree's XPU backend cannot lower because it returns early after
emitting tensor work. These tests cover the patch mechanics; the rewritten
kernel's numerical equivalence is verified on hardware, since compiling it
requires the XPU Triton backend.
"""

import sys
import types

import pytest

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import input_batch_compat


@pytest.fixture
def fake_input_batch(monkeypatch):
    """Install a stand-in vllm.v1.worker.gpu.input_batch module."""

    def original_kernel():  # pragma: no cover - never called
        raise AssertionError("the original kernel must not run")

    module = types.ModuleType("vllm.v1.worker.gpu.input_batch")
    module._combine_sampled_and_draft_tokens_kernel = original_kernel
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.input_batch", module)
    return module, original_kernel


@pytest.fixture
def stub_triton(monkeypatch):
    """Stub vllm.triton_utils so _build_kernel works without the XPU backend."""

    def jit(fn):
        return fn

    triton = types.SimpleNamespace(jit=jit)
    tl = types.SimpleNamespace(constexpr=int)
    module = types.ModuleType("vllm.triton_utils")
    module.triton = triton
    module.tl = tl
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", module)
    return module


def test_exports_public_api():
    assert callable(input_batch_compat.apply_input_batch_compat)
    assert callable(input_batch_compat._build_kernel)


def test_replaces_the_kernel(fake_input_batch, stub_triton):
    module, original = fake_input_batch

    input_batch_compat.apply_input_batch_compat()

    replaced = module._combine_sampled_and_draft_tokens_kernel
    assert replaced is not original
    assert getattr(replaced, input_batch_compat._PATCH_FLAG) is True


def test_is_idempotent(fake_input_batch, stub_triton):
    module, _ = fake_input_batch

    input_batch_compat.apply_input_batch_compat()
    first = module._combine_sampled_and_draft_tokens_kernel
    input_batch_compat.apply_input_batch_compat()

    assert module._combine_sampled_and_draft_tokens_kernel is first


def test_does_not_override_a_foreign_replacement(fake_input_batch, stub_triton):
    """A kernel someone else installed and flagged is left alone."""
    module, _ = fake_input_batch

    def foreign_kernel():  # pragma: no cover - never called
        raise AssertionError("unreachable")

    setattr(foreign_kernel, input_batch_compat._PATCH_FLAG, True)
    module._combine_sampled_and_draft_tokens_kernel = foreign_kernel

    input_batch_compat.apply_input_batch_compat()

    assert module._combine_sampled_and_draft_tokens_kernel is foreign_kernel


def test_tolerates_a_missing_kernel(fake_input_batch, stub_triton):
    """A build whose module lacks the kernel is skipped, not crashed."""
    module, _ = fake_input_batch
    del module._combine_sampled_and_draft_tokens_kernel

    input_batch_compat.apply_input_batch_compat()

    assert not hasattr(module, "_combine_sampled_and_draft_tokens_kernel")


def test_tolerates_a_build_without_the_v2_runner(monkeypatch):
    """Builds that cannot import the module are a no-op, not an error."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "vllm.v1.worker.gpu" or name.startswith(
            "vllm.v1.worker.gpu.input_batch"
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu.input_batch", raising=False)

    input_batch_compat.apply_input_batch_compat()


def test_rewritten_kernel_keeps_the_upstream_signature(fake_input_batch, stub_triton):
    """The replacement must be call-compatible with upstream's launcher."""
    import inspect

    module, _ = fake_input_batch
    input_batch_compat.apply_input_batch_compat()

    params = list(
        inspect.signature(module._combine_sampled_and_draft_tokens_kernel).parameters
    )
    assert params == [
        "input_ids_ptr",
        "idx_mapping_ptr",
        "last_sampled_tokens_ptr",
        "query_start_loc_ptr",
        "seq_lens_ptr",
        "prefill_len_ptr",
        "draft_tokens_ptr",
        "draft_tokens_stride",
        "cu_num_logits_ptr",
        "logits_indices_ptr",
        "BLOCK_SIZE",
        "NUM_NEW_SAMPLED_TOKENS",
    ]
