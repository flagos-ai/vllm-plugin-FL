# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numerics for the ``Tensor.exponential_(generator=...)`` compatibility shim.

The shim in ``vllm_fl/patches/exponential_compat.py`` has two paths:

* generator-aware: ``torch.ops.aten.exponential_(t, lambd, generator=g)``;
* vendor fallback: when the installed flag_gems registers an aten kernel
  without a ``generator`` parameter, retry without it (global RNG) and warn
  once per process.

Both paths are exercised here against real torch CPU tensors with a seeded
independent oracle (``-log(1-U)/lambd`` for the same generator state), and the
vendor-fallback path is driven by replacing ``torch.ops.aten.exponential_``
with a kernel that rejects the ``generator`` keyword.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/patches/exponential_compat.py"
_MODULE_NAME = "supp_exponential_compat_under_test"


@pytest.fixture
def compat(monkeypatch):
    """Load a fresh copy of the shim module and restore torch state after.

    Installing the shim mutates ``torch.Tensor.exponential_`` globally, and the
    module caches ``_installed``/``_warned`` flags, so every test loads its own
    module instance and the fixture restores the builtin method afterwards.
    """
    builtin_method = torch.Tensor.__dict__.get("exponential_")
    if builtin_method is None:
        # A previous shim install (e.g. another test) already shadowed it;
        # resolve the pristine descriptor from TensorBase instead.
        builtin_method = torch._C.TensorBase.exponential_

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)

    yield module

    torch.Tensor.exponential_ = builtin_method


@pytest.fixture
def vendor_kernel(monkeypatch):
    """Install an aten-style exponential_ kernel lacking ``generator``.

    Mirrors the Enflame torch_gcu bundle whose flag_gems registers
    ``aten::exponential_`` with no ``generator`` parameter: passing the keyword
    raises ``TypeError``. Records every call so tests can assert the exact
    fallback behavior.
    """
    calls = []

    def kernel(tensor, lambd=1.0):
        calls.append((float(lambd), tensor is not None))
        return tensor

    monkeypatch.setattr(torch.ops.aten, "exponential_", kernel)
    holder = type("VendorCalls", (), {"calls": calls})()
    return holder


def test_generator_path_matches_seeded_oracle(compat):
    """With a generator, the shim delegates to aten and matches -log(U)/lambd."""
    assert compat.apply_exponential_compat() is True
    assert compat.apply_exponential_compat() is False, "second install is a no-op"

    generator = torch.Generator(device="cpu").manual_seed(1234)
    drawn = torch.full((6,), 0.5, dtype=torch.float64)
    result = drawn.exponential_(lambd=1.0, generator=generator)
    assert result is drawn, "in-place exponential_ must return the same tensor"

    # Independent oracle: the same generator seed advanced by the same number
    # of uniform draws must satisfy x = -log(1 - U) / lambd elementwise (this
    # is the documented torch CPU inverse-CDF for exponential_).
    oracle_generator = torch.Generator(device="cpu").manual_seed(1234)
    uniforms = torch.rand(6, generator=oracle_generator, dtype=torch.float64)
    expected = -torch.log(1 - uniforms)
    torch.testing.assert_close(drawn, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lambd", [1.0, 2.0, 0.5])
def test_generator_path_respects_lambd(compat, lambd):
    """The lambd argument scales the seeded exponential draw exactly."""
    compat.apply_exponential_compat()

    generator = torch.Generator(device="cpu").manual_seed(99)
    drawn = torch.full((8,), 0.5, dtype=torch.float64)
    drawn.exponential_(lambd=lambd, generator=generator)

    oracle_generator = torch.Generator(device="cpu").manual_seed(99)
    uniforms = torch.rand(8, generator=oracle_generator, dtype=torch.float64)
    expected = -torch.log(1 - uniforms) / lambd
    torch.testing.assert_close(drawn, expected, rtol=1e-12, atol=1e-12)


def test_generator_kwarg_only_accepted_form(compat):
    """The generator is passed as a keyword, matching vLLM's TopKTopP usage."""
    compat.apply_exponential_compat()

    seen = {}
    original = torch.ops.aten.exponential_

    def recording_kernel(tensor, lambd=1.0, generator=None):
        seen["lambd"] = lambd
        seen["generator"] = generator
        seen["is_tensor"] = isinstance(tensor, torch.Tensor)
        return tensor

    try:
        torch.ops.aten.exponential_ = recording_kernel
        tensor = torch.full((2,), 0.5, dtype=torch.float64)
        generator = torch.Generator().manual_seed(7)
        tensor.exponential_(lambd=1.0, generator=generator)
        assert seen == {
            "lambd": 1.0,
            "generator": generator,
            "is_tensor": True,
        }
    finally:
        torch.ops.aten.exponential_ = original


def test_vendor_kernel_falls_back_without_generator(
    compat, vendor_kernel, monkeypatch, caplog
):
    """A vendor kernel without ``generator`` triggers one warned fallback."""
    compat.apply_exponential_compat()

    tensor = torch.full((3,), 0.5, dtype=torch.float64)
    with caplog.at_level(logging.WARNING, logger="supp_exponential_compat_under_test"):
        tensor.exponential_(lambd=1.0, generator=torch.Generator().manual_seed(3))

    assert vendor_kernel.calls == [(1.0, True)], "exactly one fallback call"
    assert "ignores the generator" in caplog.text
    assert compat._warned is True


def test_vendor_fallback_warns_only_once(compat, vendor_kernel, caplog):
    """The fallback warning is emitted once per module, not per call."""
    compat.apply_exponential_compat()

    with caplog.at_level(logging.WARNING, logger=_MODULE_NAME):
        for _ in range(3):
            tensor = torch.full((2,), 0.5, dtype=torch.float64)
            tensor.exponential_(generator=torch.Generator().manual_seed(1))

    assert len(vendor_kernel.calls) == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "the vendor-kernel warning must fire exactly once"


def test_no_generator_call_skips_aten_generator_branch(compat, vendor_kernel):
    """Without a generator the shim calls aten directly (no try/except path)."""
    compat.apply_exponential_compat()

    tensor = torch.full((4,), 0.5, dtype=torch.float32)
    tensor.exponential_()

    assert vendor_kernel.calls == [(1.0, True)]
    assert compat._warned is False, "no warning without a generator argument"


@pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (4, 1, 2)])
def test_shim_preserves_shape_dtype_and_device(compat, shape):
    """The shim is a transparent in-place wrapper: shape/dtype/device stable."""
    compat.apply_exponential_compat()

    generator = torch.Generator().manual_seed(11)
    tensor = torch.full(shape, 0.5, dtype=torch.float32)
    result = tensor.exponential_(generator=generator)

    assert result is tensor
    assert result.shape == shape
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.all(result >= 0), "exponential draws are non-negative"
    assert torch.all(torch.isfinite(result))


def test_install_restores_between_module_instances(compat):
    """Each fixture-loaded module instance installs exactly once.

    Guards against cross-test leakage: a fresh module must install even though
    a previous instance already replaced the method, and the fixture teardown
    must restore the pristine builtin method.
    """
    assert compat.apply_exponential_compat() is True

    spec = importlib.util.spec_from_file_location(_MODULE_NAME + "_2", _SOURCE)
    second = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME + "_2"] = second
    try:
        spec.loader.exec_module(second)
        assert second._installed is False
        assert second.apply_exponential_compat() is True
    finally:
        del sys.modules[_MODULE_NAME + "_2"]
