# SPDX-License-Identifier: Apache-2.0
"""QSA selection/attention equivalence on the actual compressed-cache geometry."""

import flag_gems
import pytest
import torch

from .test_qwen38_fgv_device import _score_oracle, _selection_oracle, _sparse_oracle
from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as portable
from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as candidate

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX QSA provider"),
]


@pytest.mark.parametrize(
    "columns, rows", [(512, 8), (2048, 8), (8192, 8), (17600, 8), (2048, 129)]
)
def test_selection_attention_and_changed_input_graph(columns, rows):
    torch.manual_seed(29)
    cache_rows = 17600
    bits = 15
    logical = torch.arange(cache_rows)
    cache = torch.zeros(cache_rows, 128, dtype=torch.bfloat16)
    cache[:, :bits] = ((logical[:, None] >> torch.arange(bits)) & 1).to(torch.bfloat16)
    cache = cache.view(11, 1600, 1, 128).cuda()
    q = torch.zeros(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    q[:, :, :bits] = (2.0 ** torch.arange(bits, device="cuda")) / 32
    table = torch.stack([torch.arange(11), torch.arange(10, -1, -1)]).int().cuda()
    full = min(columns * 4 - 1, 69632)
    positions = torch.tensor(
        [full - 1, full - 2, 0, 511, full - 1, full - 1, full - 3, full - 4],
        device="cuda",
        dtype=torch.int64,
    )
    lengths = torch.tensor([full, full - 3], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 1, 0, 1, -1, 2, 0, 1], device="cuda", dtype=torch.int32)
    if rows != 8:
        q = q.repeat((rows + 7) // 8, 1, 1)[:rows]
        positions = positions.repeat((rows + 7) // 8)[:rows]
        requests = requests.repeat((rows + 7) // 8)[:rows]
    out = torch.empty(rows, 2051, device="cuda", dtype=torch.int32)

    def run():
        with flag_gems.use_gems():
            return candidate.qsa_select_paged_tokens(
                q,
                cache,
                table,
                requests,
                positions,
                lengths,
                2048,
                4,
                out,
                num_columns=columns,
            )

    def check():
        scores, _ = _score_oracle(q, cache, table, requests, positions, lengths, 4)
        expected = _selection_oracle(scores, positions, lengths, requests)
        torch.testing.assert_close(
            out.cpu().sort().values, expected.sort().values, atol=0, rtol=0
        )
        return expected

    run()
    expected = check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    requests[:2].copy_(requests[:2].flip(0))
    positions[:2].sub_(3)
    q[1].mul_(0.5)
    graph.replay()
    graph.replay()
    expected = check()
    # Exercise the actual attention consumer with both valid selection orders.
    key = torch.randn(44, 1600, 1, 256, device="cuda", dtype=torch.bfloat16) * 0.2
    value = torch.randn_like(key)
    attn_q = torch.randn(rows, 3, 256, device="cuda", dtype=torch.bfloat16) * 0.2
    raw_table = torch.stack([torch.arange(44), torch.arange(43, -1, -1)]).int().cuda()
    new = portable.qsa_sparse_paged_attention(
        attn_q, key, value, out, raw_table, requests
    )
    old_order = portable.qsa_sparse_paged_attention(
        attn_q, key, value, expected.cuda(), raw_table, requests
    )
    independent = _sparse_oracle(attn_q, key, value, out, raw_table, requests)
    torch.testing.assert_close(new, old_order, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(new.cpu(), independent, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("rows", [1, 64, 128, 2048])
def test_short_bucket_large_batch_broadcast_and_graph(rows):
    q = torch.empty(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 1600, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    requests = (torch.arange(rows, device="cuda", dtype=torch.int32) % 3) - 1
    positions = (torch.arange(rows, device="cuda", dtype=torch.int64) * 7) % 2048
    lengths = torch.tensor([2048], device="cuda", dtype=torch.int64)
    out = torch.empty(rows, 2051, device="cuda", dtype=torch.int32)

    def run():
        with flag_gems.use_gems():
            candidate.qsa_select_paged_tokens(
                q,
                cache,
                table,
                requests,
                positions,
                lengths,
                2048,
                4,
                out,
                num_columns=512,
            )

    def check():
        ids = torch.arange(2051, dtype=torch.int32)[None, :].expand(rows, -1)
        expected = torch.where(
            (ids <= positions.cpu()[:, None]) & (requests.cpu()[:, None] == 0),
            ids,
            torch.full_like(ids, -1),
        )
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for delta in (1, 7, 31):
        positions.add_(delta).remainder_(2048)
        requests.add_(2).remainder_(3).sub_(1)
        graph.replay()
        check()
