# Copyright (c) 2026 BAAI. All rights reserved.

import pytest

pytest.importorskip("torch_npu")

from vllm.v1.worker.utils import select_common_block_size

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionBackend,
)


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_rejects_blocks_smaller_than_ascend_kernel(block_size):
    assert not AscendAttentionBackend.supports_block_size(block_size)
    assert AscendAttentionBackend.get_preferred_block_size(block_size) == 128


@pytest.mark.parametrize("manager_block_size", [128, 256, 384])
def test_hybrid_blocks_use_128_token_kernel_blocks(manager_block_size):
    assert AscendAttentionBackend.supports_block_size(manager_block_size)
    assert select_common_block_size(
        manager_block_size, [AscendAttentionBackend]
    ) == 128
