# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_layout")

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode
from vllm.v1.worker.utils import KVBlockZeroer

from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
    KunlunxinAttentionBackend,
    KunlunxinPagedAttention,
)
from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
    patch_block_table_slot_mapping,
)
from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.kv_block_zero_compat import (
    apply_kv_block_zero_compat,
)


@pytest.mark.parametrize("layout", [KVCacheLayout.LHBNC, KVCacheLayout.LBHNC])
def test_cache_planes_match_native_hnd_layout(layout):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=32, dtype=torch.float16
    )
    packed = KunlunxinAttentionBackend.customize_spec(spec)
    assert packed.page_size_bytes == spec.page_size_bytes
    assert packed.num_heads == 2
    assert packed.state_content_size_bytes == 128

    if layout is KVCacheLayout.LHBNC:
        cache = torch.zeros((2, 3, 4, 64)).transpose(0, 1)
        stride = 1
    else:
        cache = torch.zeros((3, 2, 4, 64))
        stride = 2

    key, value = KunlunxinPagedAttention.split_kv_cache(cache, 2, 32)
    assert key.shape == value.shape == (3, 2, 4, 32)
    assert KunlunxinPagedAttention.physical_block_stride(key) == stride
    key[1].fill_(1)
    value[2].fill_(2)
    assert torch.all(cache[1, 0] == 1)
    assert torch.all(cache[2, 1] == 2)
    assert torch.count_nonzero(cache[0]) == 0


def test_kv_zeroer_clears_virtual_blocks_without_touching_neighbors(monkeypatch):
    monkeypatch.setattr(KVBlockZeroer, "__init__", KVBlockZeroer.__init__)
    monkeypatch.setattr(KVBlockZeroer, "zero_block_ids", KVBlockZeroer.zero_block_ids)
    apply_kv_block_zero_compat()

    # The two aliases start at the same address. The wider view must also be
    # cleared, even if the narrower view is encountered first.
    storage = torch.ones((8, 2, 4, 32))
    spec = FullAttentionSpec(
        block_size=8, num_kv_heads=1, head_size=32, dtype=torch.float32
    )
    groups = iter(
        [
            SimpleNamespace(
                kv_cache_spec=spec,
                kv_cache_group_id=0,
                layer_names=["narrow", "wide"],
                kernel_block_stride=0,
            )
        ]
    )
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=groups,
        kernel_block_sizes=[4],
        static_forward_context={
            "narrow": SimpleNamespace(kv_cache=storage[:, :1]),
            "wide": SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )
    zeroer.zero_block_ids([2, 1, 1])
    assert torch.all(storage[:2] == 1)
    assert torch.count_nonzero(storage[2:6]) == 0
    assert torch.all(storage[6:] == 1)


def test_slot_mapping_respects_dcp_and_skips_recurrent_groups(monkeypatch):
    monkeypatch.setattr(
        BlockTable, "compute_slot_mapping", BlockTable.compute_slot_mapping
    )
    patch_block_table_slot_mapping()

    slots = torch.full((8,), 99, dtype=torch.int64)
    table = SimpleNamespace(
        block_size=4,
        dcp_world_size=2,
        dcp_rank=1,
        pcp_world_size=2,
        pcp_rank=1,
        cp_kv_cache_interleave_size=1,
        slot_mapping_mode=SlotMappingMode.NONE,
        block_table=SimpleNamespace(gpu=torch.tensor([[12, 13], [20, 21]])),
        slot_mapping=SimpleNamespace(gpu=slots),
    )
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)
    positions = torch.tensor([0, 1, 3, 8, 9])
    BlockTable.compute_slot_mapping(table, 2, starts, positions)
    assert slots.tolist() == [99] * 8

    table.slot_mapping_mode = SlotMappingMode.TOKEN_TO_KV_SLOT
    BlockTable.compute_slot_mapping(table, 2, starts, positions)
    assert slots.tolist() == [-1, 48, 49, -1, 84, -1, -1, -1]
