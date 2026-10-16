import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
    _cache_block_stride,
)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("num_blocks", [1, 8])
def test_worker_local_cache_block_addressing(interleaved, num_blocks):
    # Only a worker with both AttentionSpec and MambaSpec interleaves K/V,
    # even when both workers belong to the same hybrid model.
    shape = (2, num_blocks, 2, 4, 8)
    cache = torch.arange(2 * num_blocks * 64).reshape(shape)
    if interleaved:
        cache = cache.as_strided(shape, (64, 128, 32, 8, 1))
    key, value = cache.unbind(0)
    stride = _cache_block_stride(key)
    assert stride == (2 if interleaved else 1)
    assert _cache_block_stride(value) == stride
    for block in range(num_blocks):
        kernel_offset = block * stride * 64
        assert key[block].storage_offset() - key.storage_offset() == kernel_offset
        assert value[block].storage_offset() - value.storage_offset() == kernel_offset


def test_unsupported_cache_layout_fails_before_vendor_addressing():
    with pytest.raises(ValueError, match="dense HND"):
        _cache_block_stride(torch.empty(8, 2, 4, 16)[..., ::2])
    with pytest.raises(ValueError, match="block stride"):
        _cache_block_stride(torch.empty(24, 2, 4, 8)[::3])
