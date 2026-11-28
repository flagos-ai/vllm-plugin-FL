# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the Triton kernel launch override utilities.

``vllm_fl/patches/triton_kernel.py`` wraps a Triton kernel object in
``KernelLaunchMetaProxy`` so that ``module.kernel[grid](...)`` launches with
selected kwargs forced (e.g. vLLM's causal-conv kernels on GCU get tuned
``BLOCK_N``/``num_stages``). The proxy must:

* forward ``__getitem__`` launches to the real kernel with overrides applied;
* expose every other attribute of the wrapped kernel transparently;
* copy (not alias) the override mapping on the public property;
* be applied exactly once per kernel, rejecting conflicting re-application.
"""

from types import SimpleNamespace

import pytest

from vllm_fl.patches.triton_kernel import (
    KernelLaunchMetaProxy,
    patch_kernel_launch_meta,
)


class FakeTritonKernel:
    """Minimal Triton JIT-kernel-like object with __getitem__ launch."""

    def __init__(self):
        self.launches = []
        self.marker = "kernel-object"

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
            return f"ran-{grid}"

        return launch

    def custom_attr(self):
        return "custom-attr-value"


def test_getitem_launch_applies_overrides():
    """kernel[grid](...) forwards to the wrapped kernel with overrides merged."""
    kernel = FakeTritonKernel()
    proxy = KernelLaunchMetaProxy(kernel, {"BLOCK_N": 2048, "num_stages": 2})

    result = proxy[(4, 1, 1)](1.0, 2.0, num_warps=4)

    assert result == "ran-(4, 1, 1)"
    assert kernel.launches == [
        ((4, 1, 1), (1.0, 2.0), {"num_warps": 4, "BLOCK_N": 2048, "num_stages": 2})
    ]


def test_overrides_win_over_caller_kwargs():
    """A conflicting caller kwarg is replaced by the proxy override."""
    kernel = FakeTritonKernel()
    proxy = KernelLaunchMetaProxy(kernel, {"BLOCK_N": 1024})

    proxy["grid"](BLOCK_N=8, other=3)

    assert kernel.launches == [("grid", (), {"BLOCK_N": 1024, "other": 3})]


def test_attribute_access_is_transparent():
    """Non-launch attributes and methods resolve on the wrapped kernel."""
    kernel = FakeTritonKernel()
    proxy = KernelLaunchMetaProxy(kernel, {"BLOCK_N": 1})

    assert proxy.marker == "kernel-object"
    assert proxy.custom_attr() == "custom-attr-value"
    assert proxy.launches is kernel.launches


def test_launch_overrides_property_returns_a_copy():
    """Mutating the returned dict must not affect the live overrides."""
    kernel = FakeTritonKernel()
    proxy = KernelLaunchMetaProxy(kernel, {"BLOCK_N": 512})

    snapshot = proxy.launch_overrides
    snapshot["BLOCK_N"] = 999
    snapshot["injected"] = True

    assert proxy.launch_overrides == {"BLOCK_N": 512}


def test_empty_overrides_are_rejected():
    """Constructing without overrides is a hard error, not a silent no-op."""
    with pytest.raises(ValueError, match="At least one kernel launch override"):
        KernelLaunchMetaProxy(FakeTritonKernel(), {})


def test_patch_applies_proxy_once():
    """patch_kernel_launch_meta wraps the module attribute exactly once."""
    module = SimpleNamespace(kernel=FakeTritonKernel())

    patch_kernel_launch_meta(module, "kernel", {"num_warps": 8})
    first = module.kernel
    assert isinstance(first, KernelLaunchMetaProxy)

    # Re-applying the same overrides is a no-op returning the same proxy.
    patch_kernel_launch_meta(module, "kernel", {"num_warps": 8})
    assert module.kernel is first


def test_patch_rejects_conflicting_second_application():
    """Different overrides on an already-patched kernel raise RuntimeError."""
    module = SimpleNamespace(kernel=FakeTritonKernel())
    patch_kernel_launch_meta(module, "kernel", {"num_warps": 8})

    with pytest.raises(RuntimeError, match="already has launch overrides"):
        patch_kernel_launch_meta(module, "kernel", {"num_warps": 2})

    # The original overrides survive the rejected re-application.
    module.kernel["g"]()
    assert module.kernel._kernel.launches == [("g", (), {"num_warps": 8})]


def test_end_to_end_module_patch_launch_flow():
    """The full GCU-style flow: patch a module's kernel, launch, check args."""
    module = SimpleNamespace(_causal_conv1d_fwd_kernel=FakeTritonKernel())
    config = {"BLOCK_N": 2048, "num_stages": 2}

    patch_kernel_launch_meta(module, "_causal_conv1d_fwd_kernel", config)

    module._causal_conv1d_fwd_kernel[(1,)](q=1, k=2)
    kernel = module._causal_conv1d_fwd_kernel
    assert isinstance(kernel, KernelLaunchMetaProxy)
    assert kernel.launch_overrides == config
    underlying = kernel._kernel
    assert underlying.launches == [
        ((1,), (), {"q": 1, "k": 2, "BLOCK_N": 2048, "num_stages": 2})
    ]
    # The original mapping stays untouched by the property copy semantics.
    assert config == {"BLOCK_N": 2048, "num_stages": 2}
