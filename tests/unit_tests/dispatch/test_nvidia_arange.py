# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA FlagGems policy must preserve floating arange dtype inference."""

import pytest
import torch

from tests.utils.device_utils import get_backend

from vllm_fl.dispatch.config.utils import load_platform_config

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA FlagGems numerical regression"
)


@pytest.fixture(params=["nvidia", "nvidia_hopper"])
def gems_policy(request, monkeypatch):
    import flag_gems

    # A registration failure must not turn this into a native-only false pass.
    def fail_registration(error):
        raise RuntimeError(f"FlagGems registration failed: {error}")

    monkeypatch.setattr(flag_gems.runtime.error, "register_error", fail_registration)
    monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
    config = load_platform_config(request.param)
    return flag_gems, config["flagos_blacklist"]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "start,end,step",
    [(1 / 70, 1.0, 1 / 70), (-0.5, 0.5, 0.125), (0.0, 3.0, 1), (2, 11, 3)],
)
def test_default_nvidia_arange_preserves_dtype_and_values(
    gems_policy, start, end, step
):
    flag_gems, blacklist = gems_policy
    expected = torch.arange(start, end, step, device="cuda")
    with flag_gems.use_gems(exclude=blacklist):
        actual = torch.arange(start, end, step, device="cuda")
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_default_nvidia_policy_preserves_navit_position_bins(gems_policy):
    flag_gems, blacklist = gems_policy
    # MiniCPM-V 4.7 uses a 70x70 learned position table. Its real 24x40
    # patch grid failed when the floating arange boundaries became int64 zeros.
    # Include a second, padded item to exercise per-image coordinates and masks.
    sizes = torch.tensor([[24, 40], [16, 32]], device="cuda")
    patches = torch.arange(960, device="cuda").unsqueeze(0)
    widths = sizes[:, 1:2]
    heights = sizes[:, 0:1]
    fractional_h = (patches // widths) / heights
    fractional_w = (patches % widths) / widths
    valid = patches < widths * heights

    def position_ids():
        boundaries = torch.arange(1 / 70, 1.0, 1 / 70, device="cuda")
        rows = torch.bucketize(fractional_h, boundaries, right=True)
        cols = torch.bucketize(fractional_w, boundaries, right=True)
        return torch.where(valid, rows * 70 + cols, 0)

    expected = position_ids()
    with flag_gems.use_gems(exclude=blacklist):
        actual = position_ids()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0].unique().numel() > 1
    assert actual[1, 512:].count_nonzero().item() == 0
