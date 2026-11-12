# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the Triton launch-override proxy utilities.

A fake kernel object stands in for a Triton JIT function: its ``__getitem__``
returns a launcher that records the grid, positional arguments, and merged
keywords, which is exactly the contract the plugin relies on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_fl.patches.triton_kernel as patch_module


class _FakeKernel:
    """Minimal Triton-like kernel: subscription returns a recording launcher."""

    def __init__(self):
        self.grid = None
        self.calls = []
        self.cache_key = "fake-kernel-cache-key"

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.calls.append((args, dict(kwargs)))
            return ("LAUNCHED", args, kwargs)

        return launch

    def run(self, *args, **kwargs):
        return "RAN"


def test_proxy_requires_at_least_one_override():
    """An empty override mapping is rejected eagerly."""
    with pytest.raises(ValueError, match="At least one kernel launch override"):
        patch_module.KernelLaunchMetaProxy(_FakeKernel(), {})


def test_launch_overrides_property_is_a_defensive_copy():
    """Mutating the returned dict does not change the proxy state."""
    proxy = patch_module.KernelLaunchMetaProxy(
        _FakeKernel(), {"num_warps": 8, "num_stages": 3}
    )
    exposed = proxy.launch_overrides
    assert exposed == {"num_warps": 8, "num_stages": 3}
    exposed["num_warps"] = 1
    assert proxy.launch_overrides == {"num_warps": 8, "num_stages": 3}


def test_subscription_merges_overrides_and_forwards_arguments():
    """Overrides win over caller keywords; positional args are untouched."""
    kernel = _FakeKernel()
    proxy = patch_module.KernelLaunchMetaProxy(
        kernel, {"num_warps": 8, "num_stages": 3}
    )

    launcher = proxy[("grid-a", "grid-b")]
    result = launcher(1, 2, num_warps=2, debug=True)

    assert kernel.grid == ("grid-a", "grid-b")
    assert kernel.calls == [
        ((1, 2), {"num_warps": 8, "num_stages": 3, "debug": True})
    ]
    assert result == (
        "LAUNCHED",
        (1, 2),
        {"num_warps": 8, "num_stages": 3, "debug": True},
    )


def test_attribute_access_delegates_to_wrapped_kernel():
    """Non-overridden attributes resolve on the underlying kernel."""
    kernel = _FakeKernel()
    proxy = patch_module.KernelLaunchMetaProxy(kernel, {"num_warps": 4})

    assert proxy.cache_key == "fake-kernel-cache-key"
    assert proxy.run(3) == "RAN"
    with pytest.raises(AttributeError):
        proxy.__getattr__("does_not_exist")
    assert proxy._kernel is kernel


def test_patch_wraps_module_kernel_once():
    """Repeated installation with identical overrides is a no-op."""
    kernel = _FakeKernel()
    module = SimpleNamespace(fake_kernel=kernel)
    overrides = {"num_warps": 8}

    assert (
        patch_module.patch_kernel_launch_meta(module, "fake_kernel", overrides)
        is None
    )
    proxy = module.fake_kernel
    assert isinstance(proxy, patch_module.KernelLaunchMetaProxy)
    assert proxy._kernel is kernel

    patch_module.patch_kernel_launch_meta(module, "fake_kernel", overrides)
    assert module.fake_kernel is proxy


def test_second_install_with_different_overrides_raises():
    """Conflicting overrides on an already-wrapped kernel are rejected."""
    module = SimpleNamespace(fake_kernel=_FakeKernel())
    patch_module.patch_kernel_launch_meta(module, "fake_kernel", {"num_warps": 8})

    with pytest.raises(RuntimeError, match="already has launch overrides"):
        patch_module.patch_kernel_launch_meta(
            module, "fake_kernel", {"num_warps": 4}
        )


def test_patch_launch_applies_overrides_after_wrapping():
    """A patched module kernel launches with the configured overrides."""
    kernel = _FakeKernel()
    module = SimpleNamespace(fake_kernel=kernel)
    patch_module.patch_kernel_launch_meta(
        module, "fake_kernel", {"num_warps": 8, "num_ctas": 2}
    )

    result = module.fake_kernel[(4,)]("arg", debug=False)

    assert kernel.grid == (4,)
    assert kernel.calls == [
        (("arg",), {"num_warps": 8, "num_ctas": 2, "debug": False})
    ]
    assert result[0] == "LAUNCHED"
