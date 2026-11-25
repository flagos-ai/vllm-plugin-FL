# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.lightop_moe_sum import (
    _LightOpWithFlagGemsSum,
    adapt_lightop_moe_sum,
)


@pytest.mark.parametrize("m", [1, 8, 32, 64, 128, 256])
def test_existing_flaggems_sum_graph_replay(m):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(916)
    x = torch.randn(m, 6, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, 4096, device="cuda", dtype=torch.bfloat16)

    def unexpected_fallback(*args):
        pytest.fail("supported decode input did not use FlagGems")

    adapter = _LightOpWithFlagGemsSum(SimpleNamespace(moe_sum=unexpected_fallback))
    adapter.moe_sum(x, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        adapter.moe_sum(x, out)
    for std in (0.001, 1.0, 1000.0):
        x.normal_(std=std)
        graph.replay()
        expected = x.float().sum(1).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=0.008)
        first = out.clone()
        graph.replay()
        assert torch.equal(first, out)


@pytest.mark.parametrize("case", ["topk8", "stride", "alias", "bias", "scale", "mask", "local_tokens", "expect_m"])
def test_unsupported_abi_keeps_native(case):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    x = torch.zeros(64, 6, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 4096, device="cuda", dtype=torch.bfloat16)
    kw = {}
    if case == "topk8":
        x = torch.empty(64, 8, 4096, device="cuda", dtype=torch.bfloat16)
    elif case == "stride":
        x = torch.empty(64, 6, 8192, device="cuda", dtype=torch.bfloat16)[..., ::2]
    elif case == "alias":
        out = x.view(-1)[:64 * 4096].view(64, 4096)
    elif case == "bias":
        kw["bias"] = torch.ones_like(out)
    elif case == "scale":
        kw["factor"] = 2.0
    elif case == "mask":
        kw["expert_mask"] = torch.zeros(64, 6, device="cuda", dtype=torch.int32)
    elif case == "local_tokens":
        kw["num_local_tokens"] = torch.tensor(64, device="cuda")
    elif case == "expect_m":
        kw["expect_m"] = 64
    calls = []
    native = SimpleNamespace(moe_sum=lambda *args: calls.append(args))
    _LightOpWithFlagGemsSum(native).moe_sum(x, out, **kw)
    assert len(calls) == 1
    assert calls[0][0] is x and calls[0][1] is out
    assert calls[0][2] is kw.get("bias")
    assert calls[0][5] == kw.get("factor", 1.0)


@pytest.mark.parametrize("m", [257, 1024, 4096, 8131, 8192, 8193])
def test_prefill_sum_native_order_graph(m):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from lightop.moe import fused_experts_impl_int8_marlin

    native = fused_experts_impl_int8_marlin.__globals__["op"]
    x = torch.randn(m, 6, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, 4096, device="cuda", dtype=torch.bfloat16)
    ref = torch.empty_like(out)

    def unexpected_fallback(*args):
        pytest.fail("supported prefill did not use optimized sum")

    adapter = _LightOpWithFlagGemsSum(SimpleNamespace(moe_sum=unexpected_fallback))
    adapter.moe_sum(x, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        adapter.moe_sum(x, out)
    for std in (0.001, 1.0, 1000.0):
        x.normal_(std=std)
        graph.replay()
        native.moe_sum(x, ref)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_prefill_rollback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_MOE_SUM", "0")
    x = torch.empty(257, 6, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(257, 4096, device="cuda", dtype=torch.bfloat16)
    calls = []
    adapter = _LightOpWithFlagGemsSum(SimpleNamespace(moe_sum=lambda *a: calls.append(a)))
    adapter.moe_sum(x, out)
    assert len(calls) == 1 and calls[0][0] is x


def test_private_binding_and_rollback(monkeypatch):
    from lightop.moe import fused_experts_impl_int8_marlin as original

    native_op = original.__globals__["op"]
    adapted = adapt_lightop_moe_sum(original)
    assert adapted is not original
    assert adapted.__globals__["op"].original is native_op
    assert original.__globals__["op"] is native_op
    assert adapted.__code__ is original.__code__
    assert adapted is adapt_lightop_moe_sum(original)
    monkeypatch.setenv("VLLM_FL_HYGON_LIGHTOP_MOE_SUM", "native")
    assert adapt_lightop_moe_sum(original) is original
