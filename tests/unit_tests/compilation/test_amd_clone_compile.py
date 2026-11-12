# Copyright (c) 2026 BAAI. All rights reserved.

"""
Unit tests for AMD graph-mode clone fix (FlagGems PR #8893, commit 9a51bed).

These tests verify that flag_gems.ops.clone handles FakeTensor / meta inputs
without attempting a Triton kernel launch, which would raise:
  RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor ...)

That error was the class-1 blocker for torch.compile tracing on AMD
(flagos-ai/FlagGems#8460).  During a graph-capture warmup pass vLLM runs the
model under torch.compile; if any FlagGems op calls data_ptr() on the fake
tensor that Dynamo hands it, the warmup crashes before a single graph is
captured.

No AMD hardware required: the meta device and FakeTensorMode exercise the
same code path as a real gfx1100 graph-capture warmup.  All tests are skipped
when flag_gems is not installed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Module-level skip when flag_gems is absent (CI without FlagGems installed).
# ---------------------------------------------------------------------------
flag_gems = pytest.importorskip(
    "flag_gems",
    reason="flag_gems not installed; install >= commit 9a51bed to run",
)


# ---------------------------------------------------------------------------
# 1. New symbols from PR #8893 are present
# ---------------------------------------------------------------------------


class TestCloneGuardPresent:
    """PR #8893 adds _is_fake_or_meta and _clone_meta; both must exist."""

    def test_is_fake_or_meta_importable(self):
        from flag_gems.ops.clone import _is_fake_or_meta  # noqa: F401

    def test_clone_meta_importable(self):
        from flag_gems.ops.clone import _clone_meta  # noqa: F401

    def test_clone_impl_importable(self):
        from flag_gems.ops.clone import _clone_impl  # noqa: F401


# ---------------------------------------------------------------------------
# 2. _is_fake_or_meta correctly classifies tensors
# ---------------------------------------------------------------------------


class TestIsFakeOrMeta:
    def test_meta_tensor_true(self):
        import torch
        from flag_gems.ops.clone import _is_fake_or_meta

        t = torch.empty(4, 4, device="meta")
        assert _is_fake_or_meta(t) is True

    def test_real_cpu_tensor_false(self):
        import torch
        from flag_gems.ops.clone import _is_fake_or_meta

        t = torch.empty(4, 4, device="cpu")
        assert _is_fake_or_meta(t) is False

    def test_fake_tensor_true(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from flag_gems.ops.clone import _is_fake_or_meta

        with FakeTensorMode():
            t = torch.empty(4, 4, device="cpu")
            assert _is_fake_or_meta(t) is True


# ---------------------------------------------------------------------------
# 3. _clone_impl does not raise on meta / FakeTensor input
# ---------------------------------------------------------------------------


class TestCloneImplMetaSafety:
    """
    Before PR #8893, _clone_impl reached the Triton launcher which called
    data_ptr() on the meta tensor → RuntimeError.  The fix adds a
    _is_fake_or_meta() guard that routes to _clone_meta() instead.
    """

    def test_meta_default_format(self):
        import torch
        from flag_gems.ops.clone import _clone_impl

        t = torch.empty(8, 16, device="meta")
        out = _clone_impl(t, None)
        assert out.shape == t.shape
        assert out.device.type == "meta"

    def test_meta_contiguous_format(self):
        import torch
        from flag_gems.ops.clone import _clone_impl

        t = torch.empty(4, 8, 2, device="meta")
        out = _clone_impl(t, torch.contiguous_format)
        assert out.is_contiguous()
        assert out.shape == t.shape

    def test_meta_preserve_format(self):
        import torch
        from flag_gems.ops.clone import _clone_impl

        t = torch.empty(4, 8, device="meta")
        out = _clone_impl(t, torch.preserve_format)
        assert out.shape == t.shape

    def test_fake_tensor_no_raise(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from flag_gems.ops.clone import _clone_impl

        with FakeTensorMode():
            t = torch.empty(4, 4, device="cpu")
            out = _clone_impl(t, None)
            assert out.shape == t.shape


# ---------------------------------------------------------------------------
# 4. torch.compile fullgraph tracing over clone completes without error
# ---------------------------------------------------------------------------


class TestTorchCompileClone:
    """
    Graph capture on AMD runs the model under torch.compile in a symbolic
    tracing pass.  Before #8893 this crashed with the data_ptr() error above.
    After the fix the tracing completes and the compiled function produces
    correct output.

    backend="eager" exercises Dynamo's symbolic tracing (which hands the ops
    FakeTensors) without requiring a GPU or the ROCm Triton backend, so these
    tests are hardware-independent.  fullgraph=True ensures there are no graph
    breaks — matching the vLLM graph-capture expectation.
    """

    def test_compile_clone_default(self):
        import torch
        import flag_gems  # noqa: F401 — register ops before compile

        class M(torch.nn.Module):
            def forward(self, x):
                return x.clone()

        f = torch.compile(M(), backend="eager", fullgraph=True)
        x = torch.randn(4, 8)
        out = f(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x)

    def test_compile_clone_contiguous(self):
        import torch
        import flag_gems  # noqa: F401

        class M(torch.nn.Module):
            def forward(self, x):
                return x.clone(memory_format=torch.contiguous_format)

        f = torch.compile(M(), backend="eager", fullgraph=True)
        x = torch.randn(4, 8).t()  # deliberately non-contiguous input
        out = f(x)
        assert out.is_contiguous()
        assert out.shape == x.shape

    def test_compile_clone_preserve_format(self):
        import torch
        import flag_gems  # noqa: F401

        class M(torch.nn.Module):
            def forward(self, x):
                return x.clone(memory_format=torch.preserve_format)

        f = torch.compile(M(), backend="eager", fullgraph=True)
        x = torch.randn(4, 8)
        out = f(x)
        assert out.shape == x.shape

    def test_compile_clone_inside_larger_graph(self):
        """Clone inside a multi-op graph — matches a typical vLLM forward."""
        import torch
        import flag_gems  # noqa: F401

        class M(torch.nn.Module):
            def forward(self, x, w):
                x = x.clone()
                return torch.nn.functional.linear(x, w)

        f = torch.compile(M(), backend="eager", fullgraph=True)
        x = torch.randn(4, 8)
        w = torch.randn(16, 8)
        out = f(x, w)
        assert out.shape == (4, 16)
