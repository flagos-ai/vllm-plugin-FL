# Copyright (c) 2026 BAAI. All rights reserved.

"""input_guard contract tests for vllm_fl.ops.fla.utils.

``vllm_fl.ops.fla``'s package __init__ imports chunk.py, which needs vllm at
import time.  The tests below stub the small vllm surface chunk.py reads
before the first import; the stubs only mock the external vllm boundary,
never the module under test.
"""

import os
import sys
import types
from contextlib import nullcontext
from unittest import mock

import pytest
import torch

# chunk.py picks its implementation at import time: without flag_gems the
# flaggems branch must be off, which USE_FLAGGEMS=0 guarantees.
os.environ.setdefault("USE_FLAGGEMS", "0")


@pytest.fixture(autouse=True)
def _patch_cuda_device_ctx(monkeypatch):
    """Neutralize torch.cuda.device on CPU-only builds.

    input_guard enters ``torch.cuda.device(tensor.device.index)``; a CPU
    tensor has index None and torch raises TypeError inside ``_exchange_device``.
    ``torch.cuda.device`` is an external boundary (a torch side effect the
    guard must trigger), so it is mocked to a nullcontext here.
    """
    monkeypatch.setattr(torch.cuda, "device", lambda idx: nullcontext())


def _install_vllm_stubs():
    """Create the minimal vllm module graph vllm_fl.ops.fla imports."""
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    third_party = types.ModuleType("vllm.third_party")
    third_party.__path__ = []
    fla = types.ModuleType("vllm.third_party.flash_linear_attention")
    fla.__path__ = []
    fla_ops = types.ModuleType("vllm.third_party.flash_linear_attention.ops")
    fla_ops.__path__ = []
    l2norm = types.ModuleType("vllm.third_party.flash_linear_attention.ops.l2norm")
    l2norm.l2norm_fwd = lambda x: x
    fla_chunk = types.ModuleType("vllm.third_party.flash_linear_attention.ops.chunk")
    fla_chunk.chunk_gated_delta_rule_fwd = lambda *a, **k: None
    fla_rec = types.ModuleType(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
    )
    fla_rec.fused_recurrent_gated_delta_rule_fwd = lambda *a, **k: None
    me = types.ModuleType("vllm.model_executor")
    me.__path__ = []
    custom_op = types.ModuleType("vllm.model_executor.custom_op")

    class CustomOp:
        """Stub of vLLM's CustomOp with the decorators chunk.py applies."""

        @classmethod
        def register(cls, name):
            def decorator(obj):
                return obj

            return decorator

        @classmethod
        def register_oot(cls, _decorated_op_cls=None, name=None):
            def decorator(obj):
                return obj

            return decorator

    custom_op.CustomOp = CustomOp
    custom_op.PluggableLayer = CustomOp
    vllm.third_party = third_party
    vllm.model_executor = me
    third_party.flash_linear_attention = fla
    fla.ops = fla_ops
    fla_ops.l2norm = l2norm
    fla_ops.chunk = fla_chunk
    fla_ops.fused_recurrent = fla_rec
    me.custom_op = custom_op
    for name, mod in {
        "vllm": vllm,
        "vllm.third_party": third_party,
        "vllm.third_party.flash_linear_attention": fla,
        "vllm.third_party.flash_linear_attention.ops": fla_ops,
        "vllm.third_party.flash_linear_attention.ops.l2norm": l2norm,
        "vllm.third_party.flash_linear_attention.ops.chunk": fla_chunk,
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent": fla_rec,
        "vllm.model_executor": me,
        "vllm.model_executor.custom_op": custom_op,
    }.items():
        sys.modules[name] = mod


_STUB_ROOTS = ("vllm", "vllm_fl")

# Snapshot the pre-existing vllm/vllm_fl entries, install the stub graph,
# import the module under test, then restore: the decorator under test only
# closes over torch, so it keeps working after the stubs are gone. Without
# the restore, later files importing vllm_fl (e.g. the fused_moe suite, whose
# stub graph is keyed on "vllm_fl not yet imported") would collect-fail with
# ModuleNotFoundError: No module named 'vllm.model_executor.layers'.
_SAVED = {
    name: sys.modules.get(name, None)
    for name in list(sys.modules)
    if name.split(".")[0] in _STUB_ROOTS
}
_install_vllm_stubs()
try:
    from vllm_fl.ops.fla.utils import input_guard  # noqa: E402
finally:
    for name in [
        name
        for name in list(sys.modules)
        if name.split(".")[0] in _STUB_ROOTS and name not in _SAVED
    ]:
        sys.modules.pop(name, None)
    for name, module in _SAVED.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestInputGuard:
    """input_guard contiguity and device-context behaviour on CPU."""

    def test_non_contiguous_args_made_contiguous(self):
        """Transposed tensors are made contiguous before the call."""
        calls = []

        @input_guard
        def fn(a, b):
            calls.append((a.is_contiguous(), b.is_contiguous()))
            return a + b

        base = torch.arange(6).reshape(2, 3)
        out = fn(base.t(), base.t() + 10)
        assert calls == [(True, True)]
        torch.testing.assert_close(out, base.t() + base.t() + 10)

    def test_kwargs_tensors_also_guarded(self):
        """Keyword tensors are made contiguous too."""
        seen = {}

        @input_guard
        def fn(x=None, y=None):
            seen["x"] = x.is_contiguous()
            seen["y"] = y.is_contiguous()
            return x

        base = torch.arange(4).reshape(2, 2)
        fn(x=base.t(), y=base.t() * 2)
        assert seen == {"x": True, "y": True}

    def test_non_tensor_args_pass_through_untouched(self):
        """Non-tensor arguments are forwarded unchanged."""
        captured = {}

        @input_guard
        def fn(a, b, flag=True):
            captured.update(a=a, b=b, flag=flag)
            return a

        fn([1, 2], "text", flag=False)
        assert captured == {"a": [1, 2], "b": "text", "flag": False}

    def test_kwargs_only_tensor_still_found(self):
        """A tensor passed only by keyword still drives the device context.

        A recording ``torch.cuda.device`` double asserts the guard entered
        the device-context path for the kwargs-only tensor: the CPU tensor's
        index (None) is what reached the double, and no second context was
        opened for the non-tensor keyword.
        """
        requested = []

        def fake_device(idx):
            requested.append(idx)
            return nullcontext()

        with mock.patch.object(torch.cuda, "device", fake_device):

            @input_guard
            def fn(scale=1.0, data=None):
                return data * scale

            t = torch.ones(2, 2)
            torch.testing.assert_close(fn(data=t), t)
        # The kwargs-only tensor was found: its device index (None on CPU)
        # opened exactly one device context.
        assert requested == [None]

    def test_no_tensors_uses_nullcontext(self):
        """With no tensor anywhere the guard still runs the function."""

        @input_guard
        def fn(a, b=2):
            return a * b

        assert fn(21) == 42
        assert fn(21, b=3) == 63

    def test_wrapper_preserves_metadata(self):
        """functools.wraps keeps the original name and docstring."""

        @input_guard
        def documented_fn(a):
            """Docstring survives."""

            return a

        assert documented_fn.__name__ == "documented_fn"
        assert documented_fn.__doc__ == "Docstring survives."

    def test_result_values_unaffected_by_guard(self):
        """Guarded arithmetic equals unguarded arithmetic on contiguous input."""
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        @input_guard
        def add(x, y):
            return x + y

        torch.testing.assert_close(add(a, b), a + b, rtol=1e-6, atol=1e-7)
