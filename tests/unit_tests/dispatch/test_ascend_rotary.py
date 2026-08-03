# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the Ascend rotary embedding implementation."""

import torch

from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch
from vllm_fl.dispatch.backends.vendor.ascend.impl.rotary import rotary_embedding_ascend


def test_ascend_rotary_falls_back_for_float32():
    num_tokens = 4
    num_heads = 2
    head_size = 8

    query = torch.randn(num_tokens, num_heads, head_size)
    key = torch.randn_like(query)
    positions = torch.arange(num_tokens)
    frequencies = torch.randn(16, head_size // 2)
    cos = frequencies.cos()
    sin = frequencies.sin()

    actual_query, actual_key = rotary_embedding_ascend(
        None, query, key, cos, sin, positions, inplace=False
    )
    expected_query, expected_key = rotary_embedding_torch(
        None, query, key, cos, sin, positions, inplace=False
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
