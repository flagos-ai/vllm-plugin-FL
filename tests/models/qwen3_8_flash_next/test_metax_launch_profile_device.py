# SPDX-License-Identifier: Apache-2.0
"""Native-kernel launch presets: numerical and changed-input graph contracts."""

import os

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip("requires an assigned GPU", allow_module_level=True)

import flag_gems
import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next.launch_profile import (
    install_qwen38_launch_profile,
)


@pytest.mark.parametrize("rows", [1, 8, 64, 1024])
def test_mean_launch_presets_preserve_reference(rows):
    install_qwen38_launch_profile()
    torch.manual_seed(rows)
    for width in (128, 256, 2560):
        x = torch.randn(rows, width, device="cuda", dtype=torch.float32)
        actual = flag_gems.mean_dim(x, [-1])
        expected = x.cpu().mean(-1)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    x = torch.randn(rows, 4, 2560, device="cuda", dtype=torch.bfloat16)
    actual = flag_gems.mean_dim(x, [1])
    expected = x.cpu().float().mean(1).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
