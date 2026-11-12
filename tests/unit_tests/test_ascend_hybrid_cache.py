# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_interface")
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_fl.kv_cache.ascend.hybrid import (
    dense_attention_view,
    dense_mamba_views,
    pad_attention_specs,
)


@pytest.mark.parametrize("existing_padding", [None, 8192, 16384])
def test_attention_only_stage_uses_global_hybrid_page_size(existing_padding):
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=existing_padding,
    )
    # The final PP stage can own only attention layers. Its page size must
    # still match the global Mamba padding, without shrinking larger specs.
    padded = pad_attention_specs({"last_layer": spec}, 8192)["last_layer"]
    assert padded.page_size_bytes == max(spec.page_size_bytes, 8192)
    assert spec.page_size_padded == existing_padding


@pytest.mark.parametrize("num_blocks", [1, 3])
def test_dense_state_views_share_attention_key_pages(num_blocks):
    shapes = ((2, 3), (4, 4))
    dtypes = (torch.float16, torch.float32)
    conv_bytes, state_bytes = 12, 64
    page_bytes = conv_bytes + 2 * state_bytes
    raw = torch.zeros(num_blocks * page_bytes, dtype=torch.uint8)
    conv, state = dense_mamba_views(raw, shapes, dtypes, num_blocks)
    spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=8,
        head_size=4,
        dtype=torch.float16,
        page_size_padded=page_bytes,
    )
    attention = dense_attention_view(raw, spec, (2, num_blocks, 1, 8, 4))
    assert conv.data_ptr() == raw.data_ptr()
    assert state.data_ptr() == attention[0].data_ptr()
    assert state.data_ptr() == raw.data_ptr() + num_blocks * conv_bytes
    assert attention[1].data_ptr() == state.data_ptr() + num_blocks * state_bytes
    state.fill_(2)
    torch.testing.assert_close(
        attention[0].view(torch.float32),
        state.view_as(attention[0].view(torch.float32)),
    )
    assert torch.count_nonzero(conv) == 0
    assert torch.count_nonzero(attention[1]) == 0


def test_gdn_binds_raw_pages_as_dense_states():
    pytest.importorskip("torch_npu")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import (
        AscendQwenGatedDeltaNetAttention,
    )

    layer = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=True, hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
        ),
        get_state_shape=lambda: ((2, 3), (4, 4)),
        get_state_dtype=lambda: (torch.float16, torch.float32),
    )
    pages = torch.zeros(3, 1, 1, 140, dtype=torch.uint8)
    AscendQwenGatedDeltaNetAttention.bind_kv_cache(layer, pages)
    assert isinstance(layer.kv_cache, tuple)
    assert [tuple(s.shape) for s in layer.kv_cache] == [(3, 2, 3), (3, 4, 4)]
    assert layer.kv_cache[1].data_ptr() == pages.data_ptr() + 36
