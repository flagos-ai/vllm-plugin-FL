# Copyright (c) 2026 BAAI. All rights reserved.
"""Cache copy/zero semantics under the actual Ascend FlagGems policy."""

import pytest
import torch

pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

from vllm_fl.kv_cache.ascend.hybrid import copy_block_planes, zero_block_planes
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


def _planes(raw, layout):
    if layout == "dense":
        return [
            raw[:48].view(4, 12),
            raw[48:176].view(4, 32),
            raw[176:].view(4, 32),
        ]
    if layout == "unaligned":
        return [raw[3:31].view(4, 7)]
    if layout == "overlapping":
        return [raw[:, :3], raw[:, 2:]]
    return [raw]


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["plain", "dense", "unaligned", "overlapping"])
def test_default_ascend_policy_preserves_cache_copy_and_zero(layout):
    if layout == "dense":
        original = torch.arange(304, dtype=torch.int32).remainder(251).to(torch.uint8)
    elif layout == "unaligned":
        original = torch.arange(40, dtype=torch.uint8)
    else:
        original = torch.arange(20, dtype=torch.int32).view(4, 5)
    expected = original.clone()
    expected_planes = _planes(expected, layout)
    snapshots = [plane[[0, 1]].clone() for plane in expected_planes]
    for plane, snapshot in zip(expected_planes, snapshots):
        plane[1] = snapshot[0]
        plane[2] = snapshot[1]
        plane[3] = 0

    actual = original.npu()
    pointer = actual.data_ptr()
    planes = _planes(actual, layout)
    if layout == "dense":
        # State/K dtype views must neither duplicate byte writes nor detach
        # from the shared allocation.
        planes += [planes[1].view(torch.float32), planes[1].view(torch.float16)]
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(include=whitelist, exclude=blacklist):
        copy_block_planes(planes, [(0, 1), (1, 2)])
        zero_block_planes(planes, [3])
    torch.npu.synchronize()
    assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
