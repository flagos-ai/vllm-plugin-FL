# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
from _qwen38_cpu import ROOT, declarations

width = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py",
    "qwen38_width_cpu",
    namespace={"PreparedQSAForwardMetadata": NS},
    names={
        "qsa_column_buckets",
        "qsa_score_columns",
        "qsa_graph_columns",
        "qsa_cudagraph_key",
    },
)


@pytest.mark.parametrize(
    ("length", "columns"),
    [
        (0, 512),
        (1024, 512),
        (2051, 512),
        (2052, 2048),
        (8195, 2048),
        (8196, 8192),
        (32771, 8192),
        (32772, 17600),
        (69632, 17600),
    ],
)
def test_complete_group_boundaries(length, columns):
    assert width.qsa_score_columns(length, 4, 17600) == columns


def test_capacity_guard_and_page_alignment():
    assert width.qsa_column_buckets(17600) == (512, 2048, 8192, 17600)
    assert width.qsa_column_buckets(1600) == (512, 1600)
    with pytest.raises(ValueError, match="capacity"):
        width.qsa_score_columns(70404, 4, 17600)
    for length in range(0, 70401, 127):
        assert length // 4 <= width.qsa_score_columns(length, 4, 17600) <= 17600


def test_graph_key_separates_lengths_without_changing_token_descriptor():
    descriptor = (8, 4, True)
    one = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=512)})
    two = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=2048)})
    assert width.qsa_cudagraph_key(descriptor, one) != width.qsa_cudagraph_key(
        descriptor, two
    )
    assert width.qsa_cudagraph_key(descriptor, NS(attn_metadata={})) == descriptor
    assert width.qsa_cudagraph_key(descriptor, NS(attn_metadata=None)) == descriptor


def test_graph_key_rejects_inconsistent_layer_widths():
    with pytest.raises(ValueError, match="disagree"):
        width.qsa_graph_columns(
            {
                "a": NS(compress_ratio=4, score_columns=512),
                "b": NS(compress_ratio=4, score_columns=2048),
            }
        )
