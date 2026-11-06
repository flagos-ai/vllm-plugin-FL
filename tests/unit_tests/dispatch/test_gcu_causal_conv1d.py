# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

from vllm_fl.dispatch.backends.vendor.gcu.impl import causal_conv1d
from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy


class _FakeKernel:
    marker = "wrapped"

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return grid, args, kwargs

        return launch


def test_gcu_causal_conv1d_launch_configs():
    assert causal_conv1d.CAUSAL_CONV1D_FWD_CONFIG == {
        "BLOCK_N": 2048,
        "num_stages": 2,
    }
    assert causal_conv1d.CAUSAL_CONV1D_UPDATE_CONFIG == {"BLOCK_N": 1024}


def test_apply_causal_conv1d_gcu_patch_is_idempotent(monkeypatch):
    module = SimpleNamespace(
        _causal_conv1d_fwd_kernel=_FakeKernel(),
        _causal_conv1d_update_kernel=_FakeKernel(),
    )
    monkeypatch.setattr(causal_conv1d.importlib, "import_module", lambda _: module)

    causal_conv1d.apply_causal_conv1d_gcu_patch()
    fwd_proxy = module._causal_conv1d_fwd_kernel
    update_proxy = module._causal_conv1d_update_kernel
    causal_conv1d.apply_causal_conv1d_gcu_patch()

    assert module._causal_conv1d_fwd_kernel is fwd_proxy
    assert module._causal_conv1d_update_kernel is update_proxy
    assert isinstance(fwd_proxy, KernelLaunchMetaProxy)
    assert isinstance(update_proxy, KernelLaunchMetaProxy)
    assert fwd_proxy.launch_overrides == {
        "BLOCK_N": 2048,
        "num_stages": 2,
    }
    assert update_proxy.launch_overrides == {"BLOCK_N": 1024}
