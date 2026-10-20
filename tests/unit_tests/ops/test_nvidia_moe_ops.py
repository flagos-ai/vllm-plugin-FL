# Copyright (c) 2026 BAAI. All rights reserved.

"""Verify EMPTY padded MoE routing with independent CPU probabilities/order."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA MoE routing")


@pytest.fixture
def moe_op():
    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops.nvidia_moe_ops import register_nvidia_moe_ops

    if has_native_vllm_kernels():
        pytest.skip("Native wheel supplies its own MoE router")
    assert register_nvidia_moe_ops()
    return torch.ops._moe_C.topk_softmax


def _reference(logits, topk, renormalize, padding, bias=None):
    scores = logits.cpu().float().softmax(-1)
    choice = scores if bias is None else scores + bias.cpu()
    # Sort Python expert IDs, independently of CUDA/Gems selection kernels.
    ids = torch.tensor(
        [
            sorted(range(len(row)), key=lambda i: (-row[i], i))[:topk]
            for row in choice.tolist()
        ],
        dtype=torch.int64,
    )
    weights = scores.gather(1, ids)
    if renormalize:
        weights /= weights.sum(-1, keepdim=True)
    if padding is not None:
        ids[padding.cpu()] = -1
    source_rows = torch.tensor(
        [[k * len(logits) + row for k in range(topk)] for row in range(len(logits))],
        dtype=torch.int32,
    )
    return weights, ids, source_rows


def _outputs(rows, topk, indices_dtype):
    return (
        torch.empty(rows, topk, device="cuda", dtype=torch.float32),
        torch.empty(rows, topk, device="cuda", dtype=indices_dtype),
        torch.empty(rows, topk, device="cuda", dtype=torch.int32),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64, torch.uint32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_qwen38_padded_softmax(moe_op, dtype, indices_dtype, renormalize):
    torch.manual_seed(428)
    logits = torch.randn(17, 512, device="cuda", dtype=dtype)
    padding = torch.tensor([False, True] * 8 + [False], device="cuda")
    outputs = _outputs(17, 10, indices_dtype)
    moe_op(*outputs, logits, renormalize, None, padding)
    expected = _reference(logits, 10, renormalize, padding)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(
            actual.cpu(), reference.to(actual.dtype), rtol=2e-5, atol=2e-6
        )


@pytest.mark.gpu
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_moe_ties_bias_and_source_rows(moe_op, bias_enabled):
    logits = torch.tensor(
        [[0.0] * 8, [0.0, 1.0, 1.0, -1.0, 2.0, 2.0, 0.5, 0.0]], device="cuda"
    )
    bias = (
        torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
        if bias_enabled
        else None
    )
    outputs = _outputs(2, 3, torch.int32)
    moe_op(*outputs, logits, True, bias, None)
    for actual, reference in zip(outputs, _reference(logits, 3, True, None, bias)):
        torch.testing.assert_close(
            actual.cpu(), reference.to(actual.dtype), rtol=2e-5, atol=2e-6
        )


@pytest.mark.gpu
def test_moe_wide_router_keeps_all_experts(moe_op):
    logits = torch.linspace(-4, 4, 1536, device="cuda")[None, :].repeat(2, 1)
    padding = torch.tensor([False, True], device="cuda")
    outputs = _outputs(2, 40, torch.int64)
    moe_op(*outputs, logits, True, None, padding)
    for actual, reference in zip(outputs, _reference(logits, 40, True, padding)):
        torch.testing.assert_close(
            actual.cpu(), reference.to(actual.dtype), rtol=2e-5, atol=2e-6
        )


@pytest.mark.gpu
def test_padded_router_graph_replay(moe_op):
    logits = torch.randn(8, 512, device="cuda")
    padding = torch.zeros(8, device="cuda", dtype=torch.bool)
    outputs = _outputs(8, 10, torch.int32)

    def run():
        moe_op(*outputs, logits, True, None, padding)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    logits.copy_(torch.randn_like(logits))
    padding.copy_(torch.tensor([True, False] * 4, device="cuda"))
    graph.replay()
    for actual, reference in zip(outputs, _reference(logits, 10, True, padding)):
        torch.testing.assert_close(
            actual.cpu(), reference.to(actual.dtype), rtol=2e-5, atol=2e-6
        )


@pytest.mark.gpu
def test_fl_router_preserves_upstream_padding_branch(moe_op, monkeypatch):
    import vllm.model_executor.layers.fused_moe.router.fused_topk_router as native

    import vllm_fl.ops.fused_moe.router as fl_router

    logits = torch.randn(4, 512, device="cuda")
    padding = torch.tensor([False, True, False, True], device="cuda")
    monkeypatch.setattr(native, "_get_padding_mask", lambda _: padding)
    monkeypatch.setattr(fl_router, "_get_padding_mask", lambda _: padding)
    outputs = fl_router.fused_topk(torch.zeros(4, 16, device="cuda"), logits, 10, True)
    for actual, reference in zip(outputs, _reference(logits, 10, True, padding)):
        torch.testing.assert_close(
            actual.cpu(), reference.to(actual.dtype), rtol=2e-5, atol=2e-6
        )
