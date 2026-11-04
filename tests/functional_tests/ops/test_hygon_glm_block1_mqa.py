"""Block1 index-cache write/read, independent logits oracle, and HIP Graph."""
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import (
    fp8_paged_mqa_logits_hygon, indexer_k_quant_and_cache_hygon,
)


@pytest.mark.parametrize("heads", [32, 64])
def test_block1_index_cache_logits_and_graph(heads):
    torch.manual_seed(13)
    k_cpu = torch.randn(10, 128).bfloat16()
    q_cpu = (torch.randn(2, 1, heads, 128) * .2).to(torch.float8_e4m3fn)
    w_cpu = torch.randn(2, heads) * .1
    q, k, weights = q_cpu.cuda(), k_cpu.cuda(), w_cpu.cuda()
    slots = torch.tensor([7, 2, 5, 1, 9, 8, 0, 4, 6, 3], device="cuda")
    tables_cpu = torch.tensor([[7, 2, 5, 0, 0], [8, 0, 4, 6, 3]], dtype=torch.int32)
    tables = tables_cpu.cuda()
    lengths = torch.tensor([[3], [5]], device="cuda", dtype=torch.int32)
    cache = torch.zeros(10, 1, 132, device="cuda", dtype=torch.uint8)
    indexer_k_quant_and_cache_hygon(k, cache, slots, 128, "ue8m0")
    # Oracle uses logical input keys and independently computes quantization.
    scale = torch.exp2(torch.ceil(torch.log2(k_cpu.float().abs().amax(-1).clamp_min(1e-4) / 448)))
    quant_k = (k_cpu.float() / scale[:, None]).to(torch.float8_e4m3fn).float() * scale[:, None]
    def reference(q_values):
        out = torch.full((2, 16), -torch.inf)
        for batch, ids in enumerate(([0, 1, 2], [5, 6, 7, 8, 9])):
            scores = (q_values[batch, 0].float() @ quant_k[ids].T).relu()
            out[batch, :len(ids)] = (scores * w_cpu[batch, :, None]).sum(0)
        return out
    def run(q_values):
        return fp8_paged_mqa_logits_hygon(
            q_values, cache.unsqueeze(2), weights, lengths, tables, None, 16)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(q).cpu(), reference(q_cpu), rtol=.03, atol=.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(q)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(q)
    q_new = (q_cpu.float() * .5).to(q_cpu.dtype)
    q.copy_(q_new)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), reference(q_new), rtol=.03, atol=.003)


def test_block1_reject_speculative_queries():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import fp8_block1_mqa_hygon
    with pytest.raises(ValueError, match="single-token"):
        fp8_block1_mqa_hygon(
            torch.empty(1, 2, 32, 128, device="cuda", dtype=torch.float8_e4m3fn),
            torch.empty(1, 1, 1, 132, device="cuda", dtype=torch.uint8),
            torch.empty(2, 32, device="cuda"), torch.ones(1, device="cuda", dtype=torch.int32),
            torch.zeros(1, 1, device="cuda", dtype=torch.int32), 16)
