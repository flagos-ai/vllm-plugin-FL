from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.patches import flaggems_mm_shape_aware as shape_aware


class _FakeTensor:
    def __init__(
        self,
        *,
        m=1,
        ndim=2,
        dtype=torch.bfloat16,
        device_type="cuda",
        stride=(4096, 1),
    ):
        self.shape = (m, 4096) if ndim == 2 else (m, 4096, 1)
        self.ndim = ndim
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._stride = stride

    def stride(self):
        return self._stride


class _FakeLibrary:
    def __init__(self):
        self.impl_calls = []

    def impl(
        self,
        op_name,
        fn,
        dispatch_key,
        *,
        with_keyset=False,
        allow_override=False,
    ):
        self.impl_calls.append((op_name, fn, dispatch_key, with_keyset, allow_override))


class _FakeSafeKernel:
    def __init__(self, fn):
        self.fn = fn

    def call_boxed(self, dispatch_keys, *args):
        return self.fn(dispatch_keys, *args)


def test_default_is_disabled_and_does_not_touch_torch(monkeypatch):
    monkeypatch.delenv(shape_aware.ENABLE_ENV, raising=False)
    monkeypatch.delenv(shape_aware.THRESHOLD_ENV, raising=False)
    monkeypatch.setattr(shape_aware, "_STATE", None)

    def fail_get_kernel(*args, **kwargs):
        raise AssertionError("disabled feature must not inspect dispatch state")

    monkeypatch.setattr(shape_aware.torch.library, "get_kernel", fail_get_kernel)
    assert shape_aware.apply_shape_aware_mm() is False
    assert shape_aware._STATE is None


def test_caller_default_can_enable_but_explicit_disable_wins(monkeypatch):
    monkeypatch.delenv(shape_aware.ENABLE_ENV, raising=False)
    assert shape_aware.is_shape_aware_mm_enabled(default=True) is True

    monkeypatch.setenv(shape_aware.ENABLE_ENV, "0")
    assert shape_aware.is_shape_aware_mm_enabled(default=True) is False


@pytest.mark.parametrize(
    ("whitelist", "blacklist", "expected"),
    [
        (None, None, True),
        (None, ["linear"], True),
        (None, ["mm"], False),
        (["mm", "rms_norm"], None, True),
        (["linear"], None, False),
        ([], ["mm"], False),
    ],
)
def test_mm_dispatch_guard_preserves_explicit_flaggems_selection(
    whitelist, blacklist, expected
):
    assert shape_aware.is_mm_dispatch_enabled(whitelist, blacklist) is expected


@pytest.mark.parametrize("value", ["", "2 ", " 2", "+2", "-1", "1.0", "abc"])
def test_threshold_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, value)
    with pytest.raises(ValueError, match=shape_aware.THRESHOLD_ENV):
        shape_aware._parse_threshold_env()


def test_default_threshold_covers_running_64_and_128_is_configurable(monkeypatch):
    monkeypatch.delenv(shape_aware.THRESHOLD_ENV, raising=False)
    assert shape_aware._parse_threshold_env() == 64

    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, "128")
    assert shape_aware._parse_threshold_env() == 128


@pytest.mark.parametrize("value", ["0", "false"])
def test_disable_values_are_strict_but_supported(monkeypatch, value):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, value)
    assert shape_aware._parse_bool_env(shape_aware.ENABLE_ENV) is False


def test_enable_value_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "TRUE")
    with pytest.raises(ValueError, match=shape_aware.ENABLE_ENV):
        shape_aware._parse_bool_env(shape_aware.ENABLE_ENV)


def test_native_candidate_boundary_dtype_and_stride():
    a = _FakeTensor(m=1, stride=(4096, 1))
    # A transposed/column-major weight is a normal vLLM linear layout.
    b_column_major = _FakeTensor(m=4096, stride=(1, 4096))
    assert shape_aware._is_native_candidate(a, b_column_major, 1)

    assert shape_aware._is_native_candidate(_FakeTensor(m=64), b_column_major, 64)
    assert not shape_aware._is_native_candidate(_FakeTensor(m=65), b_column_major, 64)

    assert not shape_aware._is_native_candidate(_FakeTensor(m=2), b_column_major, 1)
    assert not shape_aware._is_native_candidate(
        a, _FakeTensor(dtype=torch.float16, stride=(1, 4096)), 1
    )
    assert not shape_aware._is_native_candidate(a, _FakeTensor(stride=(8192, 2)), 1)
    assert not shape_aware._is_native_candidate(
        _FakeTensor(stride=(8192, 2)), b_column_major, 1
    )
    assert not shape_aware._is_native_candidate(
        _FakeTensor(device_type="cpu"), b_column_major, 1
    )
    assert not shape_aware._is_native_candidate(
        _FakeTensor(ndim=3), _FakeTensor(ndim=3), 1
    )


def test_apply_captures_flaggems_before_override_and_routes_shapes(monkeypatch):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "1")
    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, "2")
    monkeypatch.setattr(shape_aware, "_STATE", None)

    calls = []

    def flaggems_mm(a, b):
        calls.append(("flaggems", a, b))
        return "flaggems-result"

    def native_mm(keyset, a, b):
        calls.append(("native", keyset, a, b))
        return "native-result"

    native_kernel = _FakeSafeKernel(native_mm)
    flaggems_kernel = _FakeSafeKernel(lambda dispatch_keys, a, b: flaggems_mm(a, b))
    library = _FakeLibrary()
    monkeypatch.setattr(
        shape_aware.torch.library,
        "get_kernel",
        lambda op_name, dispatch_key: flaggems_kernel,
    )
    monkeypatch.setattr(
        shape_aware.torch.library,
        "Library",
        lambda namespace, kind: library,
    )

    assert shape_aware.apply_shape_aware_mm(native_mm_kernel=native_kernel) is True
    assert len(library.impl_calls) == 1
    op_name, wrapper, dispatch_key, with_keyset, allow_override = library.impl_calls[0]
    assert (op_name, dispatch_key, with_keyset, allow_override) == (
        "mm",
        "CUDA",
        True,
        True,
    )
    assert wrapper._vllm_fl_original_flaggems_mm is flaggems_kernel
    assert wrapper._vllm_fl_native_mm is native_kernel

    a = _FakeTensor(m=1)
    b = _FakeTensor(m=4096, stride=(1, 4096))
    assert wrapper("dispatch-key", a, b) == "native-result"
    assert calls[0][0] == "native"

    # Boundary is inclusive: M == threshold is still decode/native.
    calls.clear()
    assert wrapper("dispatch-key", _FakeTensor(m=2), b) == "native-result"
    assert calls[0][0] == "native"

    calls.clear()
    assert wrapper("dispatch-key", _FakeTensor(m=3), b) == "flaggems-result"
    assert calls[0][0] == "flaggems"

    # Unsupported dtype/stride remains on the captured FlagGems callable.
    calls.clear()
    assert (
        wrapper("dispatch-key", _FakeTensor(dtype=torch.int8), b) == "flaggems-result"
    )
    assert calls[0][0] == "flaggems"

    # Reapplying is idempotent and does not replace the captured kernel again.
    assert shape_aware.apply_shape_aware_mm() is False


def test_apply_fails_without_safe_override_api(monkeypatch):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "1")
    monkeypatch.delenv(shape_aware.THRESHOLD_ENV, raising=False)
    monkeypatch.setattr(shape_aware, "_STATE", None)
    safe_kernel = _FakeSafeKernel(lambda dispatch_keys, a, b: None)
    monkeypatch.setattr(shape_aware.torch.library, "get_kernel", lambda *_: safe_kernel)

    class _NoAllowOverrideLibrary:
        def impl(self, op_name, fn, dispatch_key):
            del op_name, fn, dispatch_key

    monkeypatch.setattr(
        shape_aware.torch.library,
        "Library",
        lambda namespace, kind: _NoAllowOverrideLibrary(),
    )
    with pytest.raises(RuntimeError, match="with_keyset|allow_override"):
        shape_aware.apply_shape_aware_mm(native_mm_kernel=safe_kernel)
    assert shape_aware._STATE is None


def test_apply_requires_capture_before_flaggems(monkeypatch):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "1")
    monkeypatch.setattr(shape_aware, "_STATE", None)
    with pytest.raises(RuntimeError, match="before flag_gems.enable"):
        shape_aware.apply_shape_aware_mm()
