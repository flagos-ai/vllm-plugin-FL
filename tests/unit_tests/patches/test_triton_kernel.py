# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest

from vllm_fl.patches.triton_kernel import (
    KernelLaunchMetaProxy,
    patch_kernel_launch_meta,
)


class _FakeKernel:
    marker = "wrapped"

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return grid, args, kwargs

        return launch


def test_kernel_launch_meta_proxy_overrides_multiple_parameters():
    proxy = KernelLaunchMetaProxy(
        _FakeKernel(),
        {
            "BLOCK_N": 2048,
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
            "maxnreg": 64,
        },
    )

    grid, args, kwargs = proxy["grid"](
        1,
        BLOCK_N=256,
        num_warps=8,
        other=True,
    )

    assert grid == "grid"
    assert args == (1,)
    assert kwargs == {
        "BLOCK_N": 2048,
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
        "maxnreg": 64,
        "other": True,
    }
    assert proxy.marker == "wrapped"


def test_patch_kernel_launch_meta_is_idempotent():
    module = SimpleNamespace(kernel=_FakeKernel())
    overrides = {"BLOCK_N": 2048, "num_stages": 2}

    patch_kernel_launch_meta(module, "kernel", overrides)
    proxy = module.kernel
    patch_kernel_launch_meta(module, "kernel", overrides)

    assert module.kernel is proxy


def test_patch_kernel_launch_meta_rejects_conflicting_overrides():
    module = SimpleNamespace(kernel=_FakeKernel())
    patch_kernel_launch_meta(module, "kernel", {"BLOCK_N": 2048})

    with pytest.raises(RuntimeError, match="already has launch overrides"):
        patch_kernel_launch_meta(module, "kernel", {"BLOCK_N": 1024})
