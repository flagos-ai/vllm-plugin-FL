# SPDX-License-Identifier: Apache-2.0
"""Preserve scheduler ownership through vLLM 0.32's shared native KV pool."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionBackend,
    AscendAttentionState,
    AscendCommonAttentionMetadata,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import (
    AscendQwenGatedDeltaNetAttention,
)


def test_native_packed_groups_keep_128_token_attention_blocks(monkeypatch):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    (layout,) = AscendAttentionBackend.supported_kv_cache_layouts()
    cache = SimpleNamespace(
        block_size=128,
        mamba_block_size=32768,
        mamba_page_size_padded=None,
        get_resolved_kv_cache_layout=lambda: layout,
    )
    config = SimpleNamespace(
        cache_config=cache,
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    PlatformFL._align_hybrid_block_size(config, AscendAttentionBackend)
    attention = AscendAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
    )
    state = MambaSpec(
        block_size=32768,
        shapes=((16, 3), (2, 256, 256)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    groups = get_kv_cache_groups(config, {"attn": attention, "gdn": state})
    attn_group = next(g for g in groups if "attn" in g.layer_names)
    spec = attn_group.kv_cache_spec
    if hasattr(spec, "kv_cache_specs"):
        spec = spec.kv_cache_specs["attn"]
    assert spec.block_size == cache.block_size == 128
    assert cache.mamba_block_size == 32768
    assert cache.mamba_page_size_padded is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_hybrid_views_preserve_block_ownership(dtype):
    spec = AscendAttentionBackend.customize_spec(
        FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=128, dtype=dtype)
    )
    shapes = ((16, 3), (2, 4, 4))
    dtypes = (dtype, torch.float32)
    state_spec = MambaSpec(block_size=128, shapes=shapes, dtypes=dtypes)
    attention_name, state_name = "model.layers.0.attn", "model.layers.1.gdn"
    stride = spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(4 * stride, [attention_name], stride, stride),
            KVCacheTensor(4 * stride, [state_name], state_spec.page_size_bytes, stride),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec([attention_name], spec),
            KVCacheGroupSpec([state_name], state_spec),
        ],
    )
    (layout,) = AscendAttentionBackend.supported_kv_cache_layouts()
    caches = allocate_kv_cache(config, torch.device("cpu"), layout, [128, 128])
    attention, state = caches[attention_name], caches[state_name]
    assert attention.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()
    assert attention.shape == (4, 2, 128, 256)
    layer = SimpleNamespace(
        get_state_shape=lambda: shapes, get_state_dtype=lambda: dtypes
    )
    AscendQwenGatedDeltaNetAttention.bind_kv_cache(layer, state)
    conv, ssm = layer.kv_cache
    assert conv.stride(0) * conv.element_size() == stride
    assert ssm.stride(0) * ssm.element_size() == stride
    # Different groups own different block IDs even though they overlay bytes.
    attention[1].fill_(7)
    conv[2].fill_(3)
    ssm[2].fill_(4)
    torch.testing.assert_close(attention[1], torch.full_like(attention[1], 7))
    torch.testing.assert_close(conv[2], torch.full_like(conv[2], 3))
    torch.testing.assert_close(ssm[2], torch.full_like(ssm[2], 4))
    assert torch.count_nonzero(attention[0]) == 0
    assert torch.count_nonzero(attention[3]) == 0


def test_attention_phase_survives_native_metadata_remapping():
    metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=5,
        block_table_tensor=torch.tensor([[1]], dtype=torch.int32),
        slot_mapping=torch.tensor([132]),
        attn_state=AscendAttentionState.DecodeOnly,
    )
    mapped_table = torch.tensor([[2, 3]], dtype=torch.int32)
    mapped = metadata.replace(block_table_tensor=mapped_table)
    assert mapped.attn_state is AscendAttentionState.DecodeOnly
    assert mapped.block_table_tensor is mapped_table
    assert metadata.block_table_tensor.tolist() == [[1]]
