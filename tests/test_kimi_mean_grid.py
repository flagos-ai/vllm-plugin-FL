# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Grid policy and delegation contracts without Torch/CUDA imports."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_FILE = Path(__file__).parents[1] / "vllm_fl/patches/kimi_mean_grid.py"
_SPEC = importlib.util.spec_from_file_location("kimi_mean_grid_under_test", _FILE)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


def tensor(shape, dtype="bf16"):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


@pytest.mark.parametrize(
    "shape,dim,dtype,expected",
    [
        ((1, 65535, 1024), 0, "bf16", False),
        ((1, 65536, 1024), 0, "bf16", True),
        ((2, 65537, 1024), -3, "fp16", True),
        ((1, 131, 2, 131, 2, 1024), [0], "bf16", True),
        ((1, 65536, 1024), 0, "fp32", False),
        ((1, 65536, 1024), [0, 1], "bf16", False),
        ((1, 65536, 1024), [], "bf16", False),
        ((1, 65536, 1024), None, "bf16", False),
        ((1, 65536, 1024), 3, "bf16", False),
        ((0, 65536, 1024), 0, "bf16", False),
    ],
)
def test_exact_vector_grid_boundary_and_other_paths(shape, dim, dtype, expected):
    assert (
        compat._oversized_grid_y(tensor(shape, dtype), dim, ("bf16", "fp16"))
        is expected
    )


@pytest.mark.parametrize("large", [False, True])
def test_delegates_dtype_keepdim_and_identity(large):
    calls = []
    result = object()
    inp = tensor((2, 65536 if large else 32, 1024))

    def gems(*args, **kwargs):
        calls.append(("gems", args, kwargs))
        return result

    def native(*args, **kwargs):
        calls.append(("native", args, kwargs))
        return result

    wrapped = compat._make_mean_dispatch(
        gems, SimpleNamespace(call_boxed=native), "keys", ("bf16", "fp16")
    )
    assert wrapped(inp, -3, True, dtype="fp32") is result
    assert calls == [
        ("native", ("keys", inp, [-3], True), {"dtype": "fp32"})
        if large
        else ("gems", (inp, -3, True), {"dtype": "fp32"})
    ]


def test_native_error_is_not_swallowed():
    def fail(*args, **kwargs):
        raise RuntimeError("native mean validation")

    wrapped = compat._make_mean_dispatch(
        lambda *args: pytest.fail("unexpected Gems fallback"),
        SimpleNamespace(call_boxed=fail),
        "keys",
        ("bf16",),
    )
    with pytest.raises(RuntimeError, match="native mean validation"):
        wrapped(tensor((1, 65536, 1024)), 0)


def test_gate_does_not_import_runtime():
    config = SimpleNamespace(architecture="KimiK3ForConditionalGeneration")
    assert not compat.patch_kimi_mean_grid(config, enabled=False)
    assert not compat.patch_kimi_mean_grid(
        SimpleNamespace(architecture="Other"), enabled=True
    )


@pytest.mark.parametrize("already_enabled", [False, True])
def test_registration_guard_and_idempotence(monkeypatch, already_enabled):
    def original(*args, **kwargs):
        return None

    gems = SimpleNamespace(
        mean_dim=original,
        current_work_registrar=(
            SimpleNamespace(get_all_keys=lambda: ["mean.dim"])
            if already_enabled
            else None
        ),
    )
    torch = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        library=SimpleNamespace(
            get_kernel=lambda *args: SimpleNamespace(call_boxed=original)
        ),
        _C=SimpleNamespace(
            DispatchKeySet=lambda key: "keys", DispatchKey=SimpleNamespace(CUDA="CUDA")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    config = SimpleNamespace(architecture="KimiK3ForConditionalGeneration")
    if already_enabled:
        with pytest.raises(RuntimeError, match="before FlagGems.enable"):
            compat.patch_kimi_mean_grid(config, enabled=True)
        assert gems.mean_dim is original
    else:
        assert compat.patch_kimi_mean_grid(config, enabled=True)
        installed = gems.mean_dim
        gems.current_work_registrar = SimpleNamespace(get_all_keys=lambda: ["mean.dim"])
        assert not compat.patch_kimi_mean_grid(config, enabled=True)
        assert gems.mean_dim is installed
