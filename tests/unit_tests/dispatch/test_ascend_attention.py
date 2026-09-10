# Copyright (c) 2026 BAAI. All rights reserved.

import pytest

pytest.importorskip("torch_npu")

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.utils import select_common_block_size

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)


def test_ascend_backend_resolves_through_upstream_registry():
    backend = AttentionBackendEnum[AscendAttentionBackend.get_name()]
    assert backend.get_class() is AscendAttentionBackend


def test_upstream_creates_ascend_metadata_builder():
    from types import SimpleNamespace

    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import AttentionGroup

    spec = FullAttentionSpec(
        block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=None,
    )
    group = AttentionGroup(AscendAttentionBackend, ["attention"], spec, 0)
    group.create_metadata_builders(config, torch.device("cpu"), kernel_block_size=128)
    builder = group.get_metadata_builder()

    assert builder.kv_cache_spec.block_size == 128
    assert builder.layer_names == ["attention"]
    assert not builder.supports_update_block_table


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


@pytest.mark.parametrize("state", list(AscendAttentionState))
@pytest.mark.parametrize("kv_heads", [2, 8])
def test_fia_preserves_cache_blocks_and_model_kv_width(state, kv_heads):
    import torch

    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = torch.zeros(3, 128, kv_heads, 128)
    impl.value_cache = torch.ones_like(impl.key_cache)
    key = torch.zeros(5, kv_heads, 128)
    value = torch.ones_like(key)
    metadata = AscendMetadata(
        attn_state=state,
        seq_lens=torch.tensor([130, 3]),
        seq_lens_list=[130, 3],
        actual_seq_lengths_q=[2, 5],
        block_tables=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
    )
    k, v, block_size, block_table, lengths = impl._get_fia_params(
        key, value, metadata
    )

    assert block_size == 128
    if state == AscendAttentionState.PrefillNoCache:
        assert k is key and v is value
        assert block_table is None
        assert lengths == [2, 5]
    else:
        assert k.shape == v.shape == (3, 128, kv_heads * 128)
        assert k.data_ptr() == impl.key_cache.data_ptr()
        assert v.data_ptr() == impl.value_cache.data_ptr()
        torch.testing.assert_close(block_table, metadata.block_tables)
        assert lengths == [130, 3]
