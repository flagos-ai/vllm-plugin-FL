# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_interface")
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_fl.kv_cache.ascend.hybrid import pad_attention_specs


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
