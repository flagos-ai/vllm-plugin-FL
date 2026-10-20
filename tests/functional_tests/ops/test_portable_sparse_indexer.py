# SPDX-License-Identifier: Apache-2.0
"""Exercise the registered portable indexer across native top-k branch boundaries."""

from types import SimpleNamespace

import flaggems_vllm
import pytest
import torch

from vllm.model_executor.layers import sparse_attn_indexer as native
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from vllm_fl.attention import sparse_indexer as portable

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [1, 7, 32, 33])
@pytest.mark.parametrize("row_stride", [4096, 4099])
def test_registered_portable_topk_2048_with_refreshed_graph_inputs(
    monkeypatch, rows, row_stride
):
    torch.manual_seed(47)
    prefix = f"portable_test_{rows}_{row_stride}"
    platform = native.current_platform
    assert platform.is_cuda()

    # These entry points must never be reached by the portable provider.
    def forbidden(*args, **kwargs):
        pytest.fail("portable indexer called a native specialized top-k")

    monkeypatch.setattr(torch.ops._C, "cooperative_topk", forbidden)
    monkeypatch.setattr(torch.ops._C, "persistent_topk", forbidden)

    storage = torch.empty(rows, row_stride, device="cuda", dtype=torch.float32)
    logits = storage[:, :4096]
    # Exactly representable distinct scores avoid top-k tie/order assumptions.
    logits.copy_(torch.randperm(4096, device="cuda").float()[None].expand(rows, -1))
    seq_lens = torch.full((rows, 1), 4096, device="cuda", dtype=torch.int32)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=4096,
        slot_mapping=torch.arange(rows, device="cuda", dtype=torch.int64),
        num_decodes=rows,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=SimpleNamespace(
            decode_lens=torch.ones(rows, device="cuda", dtype=torch.int32),
            requires_padding=False,
            seq_lens=seq_lens,
            block_table=torch.zeros(rows, 256, device="cuda", dtype=torch.int32),
            schedule_metadata=None,
        ),
    )
    monkeypatch.setattr(
        portable,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={prefix: metadata}),
    )
    output = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    k_cache = SimpleNamespace(
        prefix=prefix,
        kv_cache=torch.empty(256, 16, 132, device="cuda", dtype=torch.uint8),
    )
    operations = {
        "indexer_k_quant_and_cache": flaggems_vllm.indexer_k_quant_and_cache,
        "fp8_fp4_paged_mqa_logits": lambda *args, **kwargs: logits,
        "top_k_per_row_decode": flaggems_vllm.top_k_per_row_decode,
    }
    monkeypatch.setitem(portable._LAYER_OPERATIONS, prefix, operations)
    layer = portable.PortableSparseAttnIndexer(
        k_cache, 128, "ue8m0", 2048, 128, 4096, 4096, output, operations=operations
    )
    hidden = torch.zeros(rows, 128, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(rows, 32, device="cuda", dtype=torch.bfloat16)

    def check():
        expected = logits.topk(2048, dim=-1).indices.sort(dim=-1).values
        torch.testing.assert_close(
            output.long().sort(dim=-1).values, expected, rtol=0, atol=0
        )

    for _ in range(3):
        assert layer(hidden, q, k, weights) is output
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer(hidden, q, k, weights)
    for shift in (137, 911):
        logits.copy_(logits.roll(shift, dims=-1))
        output.fill_(-777)
        graph.replay()
        check()
    assert native.current_platform is platform and platform.is_cuda()
