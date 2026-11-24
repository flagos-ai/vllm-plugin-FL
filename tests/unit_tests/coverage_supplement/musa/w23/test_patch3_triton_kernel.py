# Copyright (c) 2026 BAAI. All rights reserved.
"""Coverage for vllm_fl.patches.triton_kernel launch-override proxy."""

from types import SimpleNamespace

import pytest

from vllm_fl.patches.triton_kernel import (
    KernelLaunchMetaProxy,
    patch_kernel_launch_meta,
)


class _Kernel:
    """Minimal Triton JITFunction-like object: ``kernel[grid](*args)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, dict(kwargs)))
            return "ran"

        return launch

    def marker(self):
        return "marker"


def test_proxy_rejects_empty_overrides():
    with pytest.raises(ValueError, match="At least one kernel launch override"):
        KernelLaunchMetaProxy(object(), {})


def test_launch_overrides_property_returns_copy():
    source = {"num_warps": 4}
    proxy = KernelLaunchMetaProxy(object(), source)
    source["num_warps"] = 8
    snapshot = proxy.launch_overrides
    assert snapshot == {"num_warps": 4}
    snapshot["num_warps"] = 16
    assert proxy.launch_overrides == {"num_warps": 4}


def test_getitem_applies_fixed_overrides():
    kernel = _Kernel()
    proxy = KernelLaunchMetaProxy(kernel, {"num_warps": 8, "num_stages": 3})
    launch = proxy[(1, 2)]
    # Args flow through unchanged; fixed overrides are injected.
    assert launch(1, kw=2) == "ran"
    grid, args, kwargs = kernel.calls[0]
    assert grid == (1, 2)
    assert args == (1,)
    assert kwargs == {"kw": 2, "num_warps": 8, "num_stages": 3}

    # A caller cannot override a pinned launch parameter.
    launch(1, num_warps=1)
    assert kernel.calls[-1][2]["num_warps"] == 8


def test_getattr_delegates_to_wrapped_kernel():
    proxy = KernelLaunchMetaProxy(_Kernel(), {"x": 1})
    assert proxy.marker() == "marker"
    with pytest.raises(AttributeError):
        _ = proxy.absent


def test_patch_installs_proxy_once_and_guards_overrides():
    original = _Kernel()
    module = SimpleNamespace(kernel=original)

    patch_kernel_launch_meta(module, "kernel", {"num_warps": 4})
    proxy = module.kernel
    assert isinstance(proxy, KernelLaunchMetaProxy)
    assert proxy._kernel is original

    # Same overrides: idempotent, same proxy object retained.
    patch_kernel_launch_meta(module, "kernel", {"num_warps": 4})
    assert module.kernel is proxy

    # Conflicting overrides are rejected with the expected mapping in the text.
    with pytest.raises(RuntimeError, match="already has launch overrides"):
        patch_kernel_launch_meta(module, "kernel", {"num_warps": 8})


def test_patch_missing_kernel_attribute_raises():
    with pytest.raises(AttributeError):
        patch_kernel_launch_meta(SimpleNamespace(), "absent", {"x": 1})
