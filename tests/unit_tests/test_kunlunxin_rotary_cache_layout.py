# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.rotary import (
    rotary_embedding_kunlunxin,
)


@pytest.mark.parametrize("non_contiguous_cat", [False, True])
def test_rotary_kernel_receives_row_major_cache(monkeypatch, non_contiguous_cat):
    source = torch.arange(24, dtype=torch.float32).view(3, 8)
    cos, sin = source[:, :4], source[:, 4:]
    expected_cache = torch.cat((cos, sin), dim=-1)
    cat_result = (
        expected_cache.t().contiguous().t() if non_contiguous_cat else expected_cache
    )
    assert cat_result.is_contiguous() != non_contiguous_cat
    native_cat = torch.cat

    def cat(tensors, dim=0):
        torch.testing.assert_close(native_cat(tensors, dim=dim), expected_cache)
        return cat_result

    calls = []

    def rotary_kernel(positions, query, key, head_size, cache, is_neox):
        assert cache.is_contiguous()
        assert cache.stride() == (8, 1)
        torch.testing.assert_close(cache, expected_cache)
        assert head_size == 8
        assert is_neox
        calls.append(cache)

    monkeypatch.setattr(torch, "cat", cat)
    monkeypatch.setitem(
        sys.modules,
        "xtorch_ops",
        SimpleNamespace(rotary_embedding=rotary_kernel),
    )
    query = torch.zeros(2, 2, 8)
    key = torch.zeros(2, 1, 8)
    result_query, result_key = rotary_embedding_kunlunxin(
        None, query, key, cos, sin, torch.tensor([0, 2])
    )

    assert len(calls) == 1
    assert result_query.shape == query.shape
    assert result_key.shape == key.shape
