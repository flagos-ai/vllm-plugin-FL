# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Dispatch contracts without importing Torch or initializing CUDA."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_FILE = Path(__file__).parents[1] / "vllm_fl/patches/kimi_complex_ops.py"
_SPEC = importlib.util.spec_from_file_location("kimi_complex_cat_under_test", _FILE)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


class Tensor:
    def __init__(self, complex_dtype=False):
        self.complex_dtype = complex_dtype

    def is_complex(self):
        return self.complex_dtype


@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.parametrize("complex_dtype", [False, True])
def test_preserves_inputs_dim_and_output_identity(out_variant, complex_dtype):
    tensors = [Tensor(), Tensor(complex_dtype)]
    output = Tensor()
    result = object()
    calls = []

    def gems(*args, **kwargs):
        calls.append(("gems", args, kwargs))
        return result

    def native(*args, **kwargs):
        calls.append(("native", args, kwargs))
        return result

    kernel = SimpleNamespace(call_boxed=native)
    wrapped = compat._make_cat_dispatch(
        gems, kernel, "cuda_keys", out_variant=out_variant
    )
    actual = wrapped(tensors, -1, **({"out": output} if out_variant else {}))
    assert actual is result
    if complex_dtype:
        assert calls == [
            (
                "native",
                ("cuda_keys", tensors, -1),
                {"out": output} if out_variant else {},
            )
        ]
    else:
        assert calls == [
            ("gems", (tensors, -1), {"out": output} if out_variant else {})
        ]


def test_complex_output_uses_native_casting_contract():
    output = Tensor(True)
    received = []
    kernel = SimpleNamespace(
        call_boxed=lambda *args, **kwargs: received.append((args, kwargs)) or output
    )
    wrapped = compat._make_cat_dispatch(
        lambda *args, **kwargs: pytest.fail("complex output must use native"),
        kernel,
        "keys",
        out_variant=True,
    )
    tensors = [Tensor(), Tensor()]
    assert wrapped(tensors, out=output) is output
    assert received == [(("keys", tensors, 0), {"out": output})]


def test_native_error_propagates_without_fallback():
    def native(*args):
        raise RuntimeError("native validation")

    wrapped = compat._make_cat_dispatch(
        lambda *args: pytest.fail("must not swallow errors"),
        SimpleNamespace(call_boxed=native),
        "keys",
    )
    with pytest.raises(RuntimeError, match="native validation"):
        wrapped([Tensor(True)])


def test_scope_gate_does_not_import_runtime_dependencies():
    assert not compat.patch_kimi_complex_ops(
        SimpleNamespace(architecture="KimiK3ForConditionalGeneration"), enabled=False
    )
    assert not compat.patch_kimi_complex_ops(
        SimpleNamespace(architecture="OtherForCausalLM"), enabled=True
    )


def test_python_signatures_preserve_required_out():
    kernel = SimpleNamespace(call_boxed=lambda *args, **kwargs: None)
    default = compat._make_cat_dispatch(lambda *args: None, kernel, "keys")
    output = compat._make_cat_dispatch(
        lambda *args: None, kernel, "keys", out_variant=True
    )
    with pytest.raises(TypeError):
        default([Tensor()], out=Tensor())
    with pytest.raises(TypeError):
        output([Tensor()])


@pytest.mark.parametrize("complex_dtype", [False, True])
@pytest.mark.parametrize("operation", ["slice", "repeat"])
def test_view_arguments_and_result_are_preserved(complex_dtype, operation):
    inp = Tensor(complex_dtype)
    expected = object()
    calls = []

    def gems(*args):
        calls.append(("gems", args))
        return expected

    def native(*args):
        calls.append(("native", args))
        return expected

    kernel = SimpleNamespace(call_boxed=native)
    if operation == "slice":
        wrapped = compat._make_slice_dispatch(gems, kernel, "keys")
        assert wrapped(inp, -1, 1, 7, 2) is expected
        args = (inp, -1, 1, 7, 2)
    else:
        wrapped = compat._make_repeat_dispatch(gems, kernel, "keys")
        sizes = [2, 3]
        assert wrapped(inp, sizes) is expected
        args = (inp, sizes)
    assert calls == [("native", ("keys", *args)) if complex_dtype else ("gems", args)]


@pytest.mark.parametrize("already_enabled", [False, True])
def test_registration_order_is_fail_closed_and_install_is_idempotent(
    monkeypatch, already_enabled
):
    def original(*args, **kwargs):
        return None

    gems = SimpleNamespace(
        cat=original,
        cat_out=original,
        slice=original,
        repeat=original,
        current_work_registrar=(
            SimpleNamespace(get_all_keys=lambda: ["cat", "slice.Tensor", "repeat"])
            if already_enabled
            else None
        ),
    )
    fake_torch = SimpleNamespace(
        library=SimpleNamespace(
            get_kernel=lambda *args: SimpleNamespace(call_boxed=original)
        ),
        _C=SimpleNamespace(
            DispatchKeySet=lambda key: "keys", DispatchKey=SimpleNamespace(CUDA="CUDA")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    config = SimpleNamespace(architecture="KimiK3ForConditionalGeneration")
    if already_enabled:
        with pytest.raises(RuntimeError, match="before FlagGems.enable"):
            compat.patch_kimi_complex_ops(config, enabled=True)
        assert gems.cat is original and gems.slice is original
    else:
        assert compat.patch_kimi_complex_ops(config, enabled=True)
        wrappers = (gems.cat, gems.cat_out, gems.slice, gems.repeat)
        gems.current_work_registrar = SimpleNamespace(get_all_keys=lambda: ["cat"])
        assert not compat.patch_kimi_complex_ops(config, enabled=True)
        assert wrappers == (gems.cat, gems.cat_out, gems.slice, gems.repeat)
