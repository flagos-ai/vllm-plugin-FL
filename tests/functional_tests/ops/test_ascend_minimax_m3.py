# SPDX-License-Identifier: Apache-2.0
"""MiniMax component checks for a real Ascend NPU container."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3 import (
    MiniMaxM3Kernels,
    norm_rope,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.functional,
    pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU"),
]


def test_gemma_partial_rope_keeps_tail():
    x = torch.randn(4, 256, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, device="npu", dtype=torch.bfloat16) * 0.1
    positions = torch.tensor([0, 3, 9, 17], device="npu")
    angles = torch.randn(32, 32, device="npu")
    rotary = SimpleNamespace(
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1),
        rotary_dim=64,
    )
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    actual = norm_rope(x, norm, positions, rotary).view(4, 2, 128)
    xf = x.view(4, 2, 128).float()
    normalized = (
        xf
        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        * (weight.float() + 1)
    )
    cs = rotary.cos_sin_cache[positions].unsqueeze(1)
    left, right = normalized[..., :32].float(), normalized[..., 32:64].float()
    expected = torch.cat(
        (
            left * cs[..., :32] - right * cs[..., 32:],
            right * cs[..., :32] + left * cs[..., 32:],
            normalized[..., 64:].float(),
        ),
        -1,
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


def test_cache_padding_and_graph_replay():
    cache = torch.zeros(3, 2, 128, 2, 128, device="npu", dtype=torch.bfloat16)
    index_cache = torch.zeros(3, 128, 128, device="npu", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        kv_cache=cache,
        kv_size=256,
        idx_head_dim=128,
        indexer=SimpleNamespace(index_cache=SimpleNamespace(kv_cache=index_cache)),
    )
    slots = torch.tensor([0, 127, 128, -1], device="npu", dtype=torch.int64)
    key = torch.ones(4, 256, device="npu", dtype=torch.bfloat16)
    value, index_key = key * 2, key[:, :128].contiguous() * 3
    write = MiniMaxM3Kernels.insert_caches
    write(layer, key, value, index_key, slots, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write(layer, key, value, index_key, slots, slots)
    addresses = (cache.data_ptr(), index_cache.data_ptr())
    key.fill_(4)
    value.fill_(5)
    index_key.fill_(6)
    slots.copy_(torch.tensor([129, 255, 256, -1], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    assert addresses == (cache.data_ptr(), index_cache.data_ptr())
    for slot in (129, 255, 256):
        page, offset = divmod(slot, 128)
        for plane, expected in ((0, 4), (1, 5)):
            row = cache[page, plane, offset]
            torch.testing.assert_close(row, torch.full_like(row, expected))
        row = index_cache[page, offset]
        torch.testing.assert_close(row, torch.full_like(row, 6))
    assert torch.count_nonzero(cache[-1, :, -1]).item() == 0
    assert torch.count_nonzero(index_cache[-1, -1]).item() == 0


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_vendor_sparse_against_full_attention(monkeypatch, mode):
    from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3 import (
        AscendMiniMaxSparseImpl,
    )

    device = "npu"
    cache = torch.randn(4, 2, 128, 2, 128, device=device, dtype=torch.bfloat16)
    blocks = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
    if mode == "decode":
        query_lengths, prefixes, lengths = [1, 1], [136, 0], [137, 0]
        selected_rows = [[0, 1], [-1, -1]]
    else:
        query_lengths, prefixes, lengths = [2, 1], [130, 0], [132, 1]
        selected_rows = [[0, 1], [0, 1], [0, -1]]
    # Poison unused sequence positions with a large finite value: they must
    # never enter causal attention. Keep NaN poisoning a separate diagnostic.
    for request, length in enumerate(lengths):
        page, offset = divmod(length, 128)
        cache[request * 2 + page, :, offset:].fill_(1024)
        cache[request * 2 + page + 1 : request * 2 + 2].fill_(1024)
    tokens = sum(query_lengths)
    q = torch.randn(tokens + 1, 4 * 128, device=device, dtype=torch.bfloat16)
    selected = torch.tensor(
        [selected_rows, selected_rows], device=device, dtype=torch.int32
    )
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    if mode == "decode":
        detail = SimpleNamespace(
            block_table=blocks, seq_lens=seq_lens, decode_query_len=1
        )
        md = SimpleNamespace(
            num_decode_tokens=tokens,
            num_actual_tokens=tokens,
            num_decodes=2,
            num_prefills=0,
            decode=detail,
        )
    else:
        detail = SimpleNamespace(
            block_table=blocks,
            seq_lens=seq_lens,
            cu_seqlens_q=torch.tensor([0, 2, 3], device=device, dtype=torch.int32),
            context_lens=torch.tensor(prefixes, device=device, dtype=torch.int32),
            max_query_len=2,
        )
        md = SimpleNamespace(
            num_decode_tokens=0,
            num_actual_tokens=tokens,
            num_decodes=0,
            num_prefills=2,
            prefill=detail,
        )
    context = SimpleNamespace(attn_metadata={"layer": md})
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: context)
    impl = SimpleNamespace(num_heads=4, head_size=128, num_kv_heads=2, scale=128**-0.5)
    layer = SimpleNamespace(layer_name="layer", topk_indices_buffer=selected)
    output = torch.full_like(q, float("nan"))
    AscendMiniMaxSparseImpl.forward(impl, layer, q, cache, output)
    expected = torch.zeros_like(q).view(-1, 4, 128)
    cursor = 0
    for request, count in enumerate(query_lengths):
        keys = cache[request * 2 : request * 2 + 2, 0].reshape(-1, 2, 128)
        values = cache[request * 2 : request * 2 + 2, 1].reshape(-1, 2, 128)
        for offset in range(count):
            visible = min(prefixes[request] + offset + 1, lengths[request])
            if visible:
                k = keys[:visible].repeat_interleave(2, dim=1).float()
                v = values[:visible].repeat_interleave(2, dim=1).float()
                query = q[cursor + offset].view(1, 4, 128).float()
                scores = torch.einsum("qhd,khd->hqk", query, k) * 128**-0.5
                expected[cursor + offset] = torch.einsum(
                    "hqk,khd->qhd",
                    scores.softmax(-1),
                    v,
                )[0].to(q.dtype)
        cursor += count
    torch.testing.assert_close(output.view_as(expected), expected, atol=0.02, rtol=0.02)
    assert torch.isfinite(output).all().item()


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_vendor_index_reuses_output_and_masks_padding(monkeypatch, mode):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        minimax_m3_indexer as provider,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3 import (
        AscendMiniMaxIndexerImpl,
    )

    q = torch.randn(2, 2 * 128, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(4, 128, 128, device="npu", dtype=torch.bfloat16)
    blocks = torch.tensor([[0, 1], [2, 3]], device="npu", dtype=torch.int32)
    lengths = torch.tensor(
        [137, 0 if mode == "decode" else 17], device="npu", dtype=torch.int32
    )
    output = torch.empty(2, 2, 2, device="npu", dtype=torch.int32)
    common = dict(block_table=blocks, seq_lens=lengths, max_seq_len=256)
    if mode == "decode":
        detail = SimpleNamespace(**common, decode_query_len=1, max_decode_query_len=1)
        md = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=2,
            num_decodes=2,
            num_prefills=0,
            decode=detail,
        )
    else:
        detail = SimpleNamespace(
            **common,
            max_query_len=1,
            cu_seqlens_q=torch.tensor([0, 1, 2], device="npu", dtype=torch.int32),
            context_lens=torch.tensor([136, 16], device="npu", dtype=torch.int32),
        )
        md = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=2,
            prefill=detail,
        )
    monkeypatch.setattr(
        provider,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"index": md}),
    )
    impl = SimpleNamespace(
        index_cache=SimpleNamespace(prefix="index", kv_cache=cache),
        num_index_heads=2,
        index_head_dim=128,
        num_kv_heads=2,
        topk_blocks=2,
        init_blocks=1,
        local_blocks=1,
        topk_indices_buffer=output,
    )
    result = AscendMiniMaxIndexerImpl.forward(impl, q)[0 if mode == "decode" else 1]
    assert result.data_ptr() == output.data_ptr()
    expected_rows = [[0, 1], [-1, -1] if mode == "decode" else [0, -1]]
    expected = torch.tensor(
        [expected_rows, expected_rows], device="npu", dtype=torch.int32
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("with_residual", [False, True])
def test_gemma_residual_uses_unrounded_sum(with_residual, monkeypatch):
    from vllm_fl.models import minimax_m3_ascend as model

    norm = model.AscendGemmaRMSNorm(128).to(device="npu", dtype=torch.bfloat16)
    calls = []

    def selected_norm(value, weight, eps):
        calls.append((value, weight, eps))
        return MiniMaxM3Kernels.gemma_norm(value, weight, eps)

    norm._fl_kernels = SimpleNamespace(gemma_norm=selected_norm)

    def unexpected_selection():
        pytest.fail("Provider must be bound before forward")

    monkeypatch.setattr(model, "_get_kernels", unexpected_selection)
    with torch.no_grad():
        norm.weight.fill_(0.002)
    x = torch.randn(8, 128, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    expected_sum = x.float() + residual.float() if with_residual else x.float()
    expected = (
        expected_sum
        * torch.rsqrt(
            expected_sum.square().mean(-1, keepdim=True) + norm.variance_epsilon
        )
        * (1 + norm.weight.float())
    )
    result = norm(x, residual) if with_residual else norm(x)
    actual = result[0] if with_residual else result
    torch.testing.assert_close(actual, expected.to(x.dtype), atol=0, rtol=0)
    assert len(calls) == 1
    value, weight, eps = calls[0]
    assert weight is norm.weight and eps == norm.variance_epsilon
    assert value.dtype == (torch.float32 if with_residual else x.dtype)
    torch.testing.assert_close(value.float(), expected_sum, atol=0, rtol=0)
    if with_residual:
        torch.testing.assert_close(result[1], expected_sum.to(x.dtype), atol=0, rtol=0)


def test_provider_adapts_actual_upstream_implementation_layout():
    from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerTritonImpl
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseTritonImpl

    from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3 import (
        AscendMiniMaxIndexerImpl,
        AscendMiniMaxSparseImpl,
    )

    sparse = MiniMaxM3SparseTritonImpl.__new__(MiniMaxM3SparseTritonImpl)
    index = MiniMaxM3IndexerTritonImpl.__new__(MiniMaxM3IndexerTritonImpl)
    torch.nn.Module.__init__(index)
    cache = SimpleNamespace(dtype=torch.bfloat16)
    layer = SimpleNamespace(
        kv_cache_torch_dtype=torch.bfloat16,
        impl=sparse,
        indexer=SimpleNamespace(impl=index, index_cache=cache),
    )
    MiniMaxM3Kernels.prepare_sparse(layer)
    assert layer.impl is sparse and type(sparse) is AscendMiniMaxSparseImpl
    assert layer.indexer.impl is index and type(index) is AscendMiniMaxIndexerImpl
    assert callable(sparse.update_graph_params)
    assert callable(index.update_graph_params)


def test_biased_sigmoid_router_dispatch():
    from vllm.platforms import current_platform

    from vllm_fl.ops.fused_moe.router import (
        FusedTopKBiasRouterFL,
        GroupedTopKRouterFL,
    )

    if getattr(current_platform, "vendor_name", None) != "ascend":
        pytest.skip("requires active Ascend FL platform")
    hidden = torch.zeros(2, 16, device="npu", dtype=torch.bfloat16)
    logits = torch.tensor(
        [[-2.0, -0.5, 0.25, 1.5], [1.0, -1.0, 0.5, -0.25]],
        device="npu",
        dtype=torch.float32,
    )
    bias = torch.tensor([1.3, 0.4, -0.6, -1.0], device="npu")
    scores = logits.sigmoid()
    expected_ids = (scores + bias).topk(2, dim=-1).indices
    # Bias deliberately changes the selected expert set.
    assert not torch.equal(expected_ids, scores.topk(2, dim=-1).indices)
    for renormalize in (False, True):
        for indices_type in (None, torch.int32, torch.int64):
            # Both ordinary bias routing and invalid-group fallback share the
            # same vendor-scoped helper. No router initialization is needed.
            router = SimpleNamespace(
                top_k=2,
                renormalize=renormalize,
                scoring_func="sigmoid",
                e_score_correction_bias=bias,
                routed_scaling_factor=2.5,
                num_expert_group=4,
            )
            for cls in (FusedTopKBiasRouterFL, GroupedTopKRouterFL):
                weights, ids = cls._compute_routing(
                    router,
                    hidden,
                    logits,
                    indices_type,
                )
                assert ids.dtype == (
                    torch.int32 if indices_type is None else indices_type
                )
                assert weights.dtype == torch.float32
                torch.testing.assert_close(
                    ids.long().sort(-1).values,
                    expected_ids.sort(-1).values,
                )
                expected_weights = scores.gather(1, ids.long())
                if renormalize:
                    expected_weights /= expected_weights.sum(-1, keepdim=True)
                torch.testing.assert_close(weights, expected_weights * 2.5)


def test_new_model_namespace_uses_fl_moe_factory():
    from vllm.models.minimax_m3.nvidia import model

    from vllm_fl.ops.custom_ops import register_oot_ops
    from vllm_fl.ops.fused_moe.layer import FusedMoEFL

    register_oot_ops()
    assert model.FusedMoE is FusedMoEFL
