# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Eager DSpark DP padding must expose no stale query or cache-address rows."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from vllm_fl.models.deepseek_v41.cache import DeepseekV41CacheMetadataBuilder

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="device metadata kernels"),
]


def test_real_query_metadata_pads_and_clears_previous_batch():
    device = torch.device("cuda")
    spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
    )
    owner = SimpleNamespace(
        kind="swa", kv_cache=torch.empty((5, 64, 584), dtype=torch.uint8, device=device)
    )
    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        compilation_config=SimpleNamespace(static_forward_context={"draft": owner}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=40),
        model_config=SimpleNamespace(max_model_len=256),
    )
    builder = DeepseekV41CacheMetadataBuilder(spec, ["draft"], cfg, device)

    def build(requests, end):
        n = requests * 5
        query = torch.arange(requests + 1, dtype=torch.int32) * 5
        common = CommonAttentionMetadata(
            query_start_loc=query.to(device),
            query_start_loc_cpu=query,
            seq_lens=torch.full((requests,), end, dtype=torch.int32, device=device),
            num_reqs=requests,
            num_actual_tokens=n,
            max_query_len=5,
            max_seq_len=256,
            block_table_tensor=torch.tensor(
                [[1, 2, 3, 4]] * requests, dtype=torch.int32, device=device
            ),
            slot_mapping=torch.ones(n, dtype=torch.int64, device=device),
            causal=False,
            positions=torch.arange(
                end - 5, end, dtype=torch.int64, device=device
            ).repeat(requests),
        )
        return builder.build(0, common)

    build(4, 205)
    actual = build(1, 165).for_num_tokens(40)
    logical = torch.arange(32, 165)
    expected = (logical // 64 + 1) * 64 + logical % 64
    torch.testing.assert_close(
        actual.swa_indices[:5, 0, :133].cpu(),
        expected.to(torch.int32).expand(5, -1),
        rtol=0,
        atol=0,
    )
    assert (actual.swa_indices[:, 0, 133:] == -1).all()
    assert (actual.swa_indices[5:] == -1).all()
    assert (actual.slot_mapping[5:] == -1).all()
    assert (actual.token_to_req[5:] == -1).all()
    assert (actual.context_lens[5:] == 0).all()
    with pytest.raises(ValueError, match="capacity"):
        actual.for_num_tokens(41)


def test_causal_205_rows_after_full_batch():
    device = torch.device("cuda")
    spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
    )
    owner = SimpleNamespace(
        kind="swa", kv_cache=torch.empty((5, 64, 584), dtype=torch.uint8, device=device)
    )
    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        compilation_config=SimpleNamespace(static_forward_context={"target": owner}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        model_config=SimpleNamespace(max_model_len=256),
    )
    builder = DeepseekV41CacheMetadataBuilder(spec, ["target"], cfg, device)

    def build(n):
        q = torch.tensor([0, n], dtype=torch.int32)
        c = CommonAttentionMetadata(
            query_start_loc=q.to(device),
            query_start_loc_cpu=q,
            seq_lens=torch.tensor([n], dtype=torch.int32, device=device),
            num_reqs=1,
            num_actual_tokens=n,
            max_query_len=n,
            max_seq_len=256,
            block_table_tensor=torch.tensor(
                [[1, 2, 3, 4]], dtype=torch.int32, device=device
            ),
            slot_mapping=torch.ones(n, dtype=torch.int64, device=device),
            causal=True,
            positions=torch.arange(n, dtype=torch.int64, device=device),
        )
        return builder.build(0, c)

    build(256)
    actual = build(205).for_num_tokens(256)
    torch.testing.assert_close(
        actual.slot_mapping[:205].cpu(),
        torch.arange(64, 269, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    assert (actual.token_to_req[:205] == 0).all()
    assert (actual.slot_mapping[205:] == -1).all()
    assert (actual.token_to_req[205:] == -1).all()
    assert (actual.swa_indices[205:] == -1).all()
    assert (actual.context_lens[205:] == 0).all()
