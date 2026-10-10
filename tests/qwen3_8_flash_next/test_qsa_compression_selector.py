# SPDX-License-Identifier: Apache-2.0
"""Compression selection follows the layout supported by the fused kernel."""

from types import SimpleNamespace

import pytest

from vllm_fl.models.qwen3_8_flash_next.gpu import indexer_qsa
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa


@pytest.mark.parametrize(
    "dim,neox,fused",
    [
        (128, True, True),
        (64, True, False),
        (128, False, False),
    ],
)
def test_compression_selector_matches_supported_layout(dim, neox, fused):
    instance = indexer_qsa.QSAIndexer.__new__(indexer_qsa.QSAIndexer)
    instance.index_head_dim = dim
    instance.select_all_tokens = False
    instance.rotary_emb = SimpleNamespace(rotary_dim=64, is_neox_style=neox)
    selected = instance._compression_impl()
    assert selected is (
        qsa.qsa_compress_norm_mrope_store_groups
        if fused
        else qsa.qsa_compress_groups_with_ratio
    )
