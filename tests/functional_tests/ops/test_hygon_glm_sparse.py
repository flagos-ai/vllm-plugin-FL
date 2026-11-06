"""Small BF16 sparse MLA reference, block mapping, and HIP Graph tests."""
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import (
    hygon_glm_sparse_mla,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)


def inputs(heads=8, block_size=64):
    torch.manual_seed(42)
    q = torch.randn(3, heads, 576, device="cuda", dtype=torch.bfloat16) * .2
    kv = torch.randn(4, block_size, 576, device="cuda", dtype=torch.bfloat16) * .2
    idx = torch.full((3, 2048), -1, device="cuda", dtype=torch.int32)
    # Ragged and non-contiguous valid slots; final row is all invalid.
    idx[0, :3] = torch.tensor([0, block_size - 1, block_size], device="cuda")
    idx[0, 5] = block_size + 1
    idx[1, :2] = torch.tensor([2, block_size + 8], device="cuda")
    return q, kv, idx


def reference(q, kv, indices, scale=.125):
    flat = kv.reshape(-1, 576).float()
    rows = []
    for row, ids in zip(q.float(), indices):
        valid = ids[(ids >= 0) & (ids < flat.shape[0])].long()
        if valid.numel() == 0:
            rows.append(torch.zeros(row.shape[0], 512, device=q.device))
        else:
            selected = flat[valid]
            rows.append(((row @ selected.T) * scale).softmax(-1) @ selected[:, :512])
    return torch.stack(rows).to(q.dtype)


def mapped(q, kv, logical):
    req = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    blocks = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
    return triton_convert_req_index_to_global_index(
        req, blocks, logical, BLOCK_SIZE=kv.shape[1])


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_ragged_sparse_block_mapping(heads, block_size):
    q, kv, logical = inputs(heads, block_size)
    indices = mapped(q, kv, logical)
    assert indices[0, :3].tolist() == [2 * block_size, 3 * block_size - 1, 0]
    actual = hygon_glm_sparse_mla(q, kv, indices, .125)
    torch.testing.assert_close(actual, reference(q, kv, indices), rtol=.03, atol=.003)


def test_fake_compile_and_graph_changed_inputs():
    q, kv, idx = inputs()
    compiled = torch.compile(hygon_glm_sparse_mla, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(q, kv, idx, .125),
                               reference(q, kv, idx), rtol=.03, atol=.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            hygon_glm_sparse_mla(q, kv, idx, .125)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = hygon_glm_sparse_mla(q, kv, idx, .125)
    q.mul_(.5)
    idx[1, 0] = 100
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(q, kv, idx), rtol=.03, atol=.003)


def test_reject_fp8_cache():
    q, kv, idx = inputs()
    with pytest.raises(ValueError, match="BF16"):
        hygon_glm_sparse_mla(q, kv.to(torch.float8_e4m3fn), idx, .125)


@pytest.mark.parametrize("sq", [1, 2, 4, 8, 16, 32, 64, 65])
def test_native_customop_graph_and_index_ownership(sq):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import (
        hygon_glm_native_sparse_mla as op,
    )
    torch.manual_seed(42)
    q = torch.randn(sq, 4, 576, device="cuda", dtype=torch.bfloat16) * .2
    kv = torch.randn(256, 16, 576, device="cuda", dtype=torch.bfloat16) * .2
    idx = torch.full((sq, 2048), -1, device="cuda", dtype=torch.int32)
    idx[:, :5] = torch.tensor([4000, 1, 2049, 2049, 4096], device="cuda")
    idx[0] = -1
    before = idx.clone()
    compiled = torch.compile(op, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(q, kv, idx, .0625),
                               reference(q, kv, idx, .0625), rtol=.03, atol=.003)
    torch.testing.assert_close(idx, before, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(q, kv, idx, .0625)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(q, kv, idx, .0625)
    q.neg_()
    idx[0, :3] = torch.tensor([3999, 3, 129], device="cuda")
    if sq > 1:
        idx[1] = -1
    before = idx.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(q, kv, idx, .0625), rtol=.03, atol=.003)
    torch.testing.assert_close(idx, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="scope"):
        op(q, kv, idx, .125)


def test_native_paged_cache_integration(monkeypatch):
    from types import SimpleNamespace as NS
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import HygonGLMSparseImpl
    monkeypatch.setenv("VLLM_FL_HYGON_GLM_MLA_KERNEL", "native_flashmla")
    q, cache, idx = inputs(4, 16)
    impl = object.__new__(HygonGLMSparseImpl)
    impl.topk_indices_buffer = idx
    impl.softmax_scale = .0625
    metadata = NS(req_id_per_token=torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32),
                  block_table=torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32),
                  block_size=16)
    actual = impl.forward_mqa((q[..., :512], q[..., 512:]), cache, metadata, None)[0]
    torch.testing.assert_close(actual, reference(q, cache, mapped(q, cache, idx), .0625),
                               rtol=.03, atol=.003)


def test_metadata_without_aiter_and_block_selection():
    from types import SimpleNamespace as NS
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import (
        HygonGLMSparseBackend, HygonGLMSparseMetadataBuilder,
    )
    from vllm.v1.worker.utils import select_common_block_size
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
    for block in (16, 64):
        assert select_common_block_size(
            block, [HygonGLMSparseBackend]) == block
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == [1, 64]
    assert select_common_block_size(
        16, [HygonGLMSparseBackend, DeepseekV32IndexerBackend]) == 1
    config = NS(model_config=NS(
        hf_config=NS(model_type="glm_moe_dsa", index_topk=2048),
        dtype=torch.bfloat16, get_num_attention_heads=lambda _: 16),
        parallel_config=NS(decode_context_parallel_size=1),
        scheduler_config=NS(max_num_batched_tokens=8),
        cache_config=NS(cache_dtype="auto"))
    builder = HygonGLMSparseMetadataBuilder(NS(block_size=16), ["layer"], config,
                                           torch.device("cuda"))
    common = NS(num_actual_tokens=3, num_reqs=2, max_query_len=2, max_seq_len=16,
                query_start_loc_cpu=torch.tensor([0, 2, 3], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32),
                slot_mapping=torch.tensor([0, 1, 2], device="cuda"),
                block_table_tensor=torch.tensor([[2], [0]], device="cuda", dtype=torch.int32))
    metadata = builder.build(0, common)
    assert metadata.req_id_per_token.tolist() == [0, 0, 1]
    assert metadata.block_size == 16
    assert metadata.fp8_extra_metadata is None
    assert builder._cudagraph_support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


@pytest.mark.parametrize("block", [1, 16])
def test_cache_write_and_attention_graph(block):
    from types import SimpleNamespace as NS
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import (
        HygonGLMSparseImpl, hygon_glm_cache_write,
    )
    q, cache, idx = inputs(16, block)
    cache.zero_()
    kv = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([2 * block, -1, 4 * block - 1], device="cuda")
    scale = torch.ones(1, device="cuda")
    compiled = torch.compile(hygon_glm_cache_write, backend="eager", fullgraph=True)
    compiled(kv, rope, cache, slots, scale)
    expected = torch.zeros_like(cache).reshape(-1, 576)
    expected[slots[0]] = torch.cat((kv[0], rope[0]))
    expected[slots[2]] = torch.cat((kv[2], rope[2]))
    torch.testing.assert_close(cache.reshape(-1, 576), expected, rtol=0, atol=0)
    impl = object.__new__(HygonGLMSparseImpl)
    impl.topk_indices_buffer = idx
    impl.softmax_scale = .125
    metadata = NS(req_id_per_token=torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32),
                  block_table=torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32),
                  block_size=block)
    def run():
        impl.do_kv_cache_update(kv, rope.unsqueeze(1), cache, slots, "auto", scale)
        return impl.forward_mqa((q[..., :512], q[..., 512:]), cache, metadata, None)[0]
    expected_out = reference(q, cache, mapped(q, cache, idx))
    torch.testing.assert_close(run(), expected_out, rtol=.03, atol=.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    kv.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference(q, cache, mapped(q, cache, idx)),
                               rtol=.03, atol=.003)
