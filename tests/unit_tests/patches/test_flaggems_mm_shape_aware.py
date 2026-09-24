from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl import flaggems_runtime as runtime
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


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.setattr(runtime, "_STATE", None)
    monkeypatch.setattr(runtime, "_FAILED", False)


def test_disabled_policy_is_latched_without_kernel_inspection(monkeypatch):
    monkeypatch.delenv(shape_aware.ENABLE_ENV, raising=False)
    monkeypatch.delenv(shape_aware.THRESHOLD_ENV, raising=False)
    calls = []
    monkeypatch.setattr(
        shape_aware,
        "_get_registered_mm_kernel",
        lambda: pytest.fail("disabled policy inspected mm"),
    )
    assert runtime.configure_flaggems(calls.append).status == "disabled"
    assert runtime.configure_flaggems(calls.append).status == "disabled"
    assert calls == [None]
    monkeypatch.setenv(shape_aware.ENABLE_ENV, "1")
    with pytest.raises(RuntimeError, match="process-lifetime"):
        runtime.configure_flaggems(calls.append)


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


@pytest.mark.parametrize(
    "scenario",
    ["reuse", "disable", "threshold", "backend", "external", "filtered"],
)
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not callable(getattr(torch.library, "get_kernel", None))
    or not torch._C._dispatch_has_kernel_for_dispatch_key("aten::mm", "CUDA"),
    reason="Real CUDA dispatcher probe requires NVIDIA and SafeKernelFunction",
)
def test_real_dispatcher_process_lifetime(monkeypatch, scenario):
    # Real Library and SafeKernelFunction, isolated so aten registrations never
    # leak into other tests. CPU tensors go through a captured CUDA Python
    # implementation for the large-M branch; no CUDA tensors are allocated.
    import os
    import subprocess
    import sys
    import textwrap

    env = os.environ.copy()
    env[shape_aware.ENABLE_ENV] = "1"
    env[shape_aware.THRESHOLD_ENV] = "2"
    code = textwrap.dedent("""
        import os, sys, torch
        from vllm_fl.patches import flaggems_mm_shape_aware as m
        from vllm_fl import flaggems_runtime as runtime
        scenario = sys.argv[1]
        calls = []
        def fake_mm(a, b):
            calls.append('large')
            return a + 7
        fake_mm.__module__ = 'flag_gems.test_backend'
        def enable(lib):
            calls.append('enable')
            if scenario == 'filtered':
                return
            lib.impl('mm', fake_mm, 'CUDA')
        if scenario == 'filtered':
            try:
                runtime.configure_flaggems(enable)
            except RuntimeError as e:
                assert 'did not register' in str(e), str(e)
            else:
                raise AssertionError('accepted missing/wrong backend')
            try:
                runtime.configure_flaggems(enable)
            except RuntimeError as e:
                assert 'previously failed' in str(e)
            else:
                raise AssertionError('accepted retry after failed init')
            sys.exit(0)
        status = runtime.configure_flaggems(enable)
        assert status.status == 'installed'
        assert status.native != status.flaggems
        keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
        a, b = torch.ones(3, 2), torch.ones(2, 2)
        result = torch.library.get_kernel('aten::mm','CUDA').call_boxed(keys,a,b)
        torch.testing.assert_close(result, a + 7)
        assert calls == ['enable', 'large']
        if scenario == 'reuse':
            assert runtime.configure_flaggems(enable).status == 'already_active'
            assert calls == ['enable', 'large']
            sys.exit(0)
        kwargs = {}
        if scenario == 'disable': os.environ[m.ENABLE_ENV] = '0'
        if scenario == 'threshold': os.environ[m.THRESHOLD_ENV] = '3'
        if scenario == 'backend': kwargs['blacklist'] = ['mm']
        if scenario == 'external':
            external = torch.library.Library('aten','IMPL')
            external.impl('mm', lambda a,b: a + 100, 'CUDA', allow_override=True)
        try:
            runtime.configure_flaggems(enable, **kwargs)
        except RuntimeError as e:
            assert ('conflicting_owner' if scenario == 'external' else 'process-lifetime') in str(e)
        else:
            raise AssertionError('accepted changed configuration/owner')
        assert calls == ['enable', 'large']
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, scenario],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_apply_fails_without_safe_override_api():
    class OldLibrary:
        def impl(self, name, fn, key):
            pytest.fail("unsafe registration attempted")

    with pytest.raises(RuntimeError, match="with_keyset|allow_override"):
        shape_aware._register_override(OldLibrary(), lambda *args: None)


@pytest.mark.parametrize(
    "use_flaggems,enabled,blacklist",
    [
        (False, "invalid", None),
        (True, "0", None),
        (True, "invalid", ["mm"]),
    ],
)
def test_inactive_mm_ignores_unrelated_threshold(
    monkeypatch, use_flaggems, enabled, blacklist
):
    monkeypatch.setenv(shape_aware.ENABLE_ENV, enabled)
    monkeypatch.setenv(shape_aware.THRESHOLD_ENV, "invalid")
    calls = []
    result = runtime.configure_flaggems(
        calls.append, use_flaggems=use_flaggems, blacklist=blacklist
    )
    assert result.status == "disabled"
    assert calls == ([None] if use_flaggems else [])
