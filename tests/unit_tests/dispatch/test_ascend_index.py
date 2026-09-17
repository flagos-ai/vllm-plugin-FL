# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

from vllm_fl.utils import get_flag_gems_whitelist_blacklist


@pytest.mark.gpu
@pytest.mark.parametrize("cache_half", [0, 1])
def test_default_ascend_index_preserves_vision_rope_cache_strides(cache_half):
    # get_cos_sin returns slices of a shared [position, cos+sin] cache.
    cache = torch.arange(32 * 36, dtype=torch.float32).view(32, 36)
    indices = torch.stack([torch.arange(280) % 14, torch.arange(280) % 20], dim=1)
    reference = cache.chunk(2, dim=-1)[cache_half][indices]
    npu_cache = cache.npu().chunk(2, dim=-1)[cache_half]
    assert not npu_cache.is_contiguous()
    _, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(exclude=blacklist):
        actual = npu_cache[indices.npu()]
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
