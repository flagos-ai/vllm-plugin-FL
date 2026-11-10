# Copyright (c) 2026 BAAI. All rights reserved.
"""Packed W8A8 prefill down: dynamic routes, epilogue and dispatch isolation."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP required")
from vllm_fl.dispatch.backends.vendor.hygon.impl.moe import prefill_moe_gemm2 as impl
from lightop._lmslim_native.layers.fused_moe import int8_marlin as vendor

CONFIG = dict(BLOCK_SIZE_M=32, MODE=568, DELTA=4)

@pytest.fixture(scope="module")
def weight():
    torch.manual_seed(2097)
    raw = torch.randint(-127, 128, (256, 4096, 256), device="cuda", dtype=torch.int8)
    packed = raw.transpose(1, 2).reshape(256, 4, 64, 4096).transpose(2, 3).contiguous().reshape(256, 4, 262144)
    scale = torch.rand(256, 4096, 1, device="cuda") * 0.01
    return raw, packed, scale


def case(m, weight):
    raw, packed, sb = weight
    q = torch.randint(-127, 128, (m * 6, 256), device="cuda", dtype=torch.int8)
    sa = torch.rand(m * 6, 1, device="cuda") * 0.001
    rw = torch.rand(m, 6, device="cuda")
    ids = torch.rand(m, 256, device="cuda").topk(6, dim=1).indices.int()
    aligned = vendor.moe_align_block_size(ids, 128, 256, None)
    out = torch.empty(m, 6, 4096, device="cuda", dtype=torch.bfloat16)
    return [q, packed, out, sa, sb, rw, *aligned, 1, CONFIG.copy()], ids


@pytest.mark.parametrize("m", [4096, 8131, 8192])
def test_dynamic_graph_and_cpu_reference(m, weight):
    args, ids = case(m, weight)
    q, w, out, sa, sb, rw, si, ei, pad, _, cfg = args
    assert impl.supports_prefill_down(*args)
    impl.prefill_down(*args[:9])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.prefill_down(*args[:9])
    for kind in ("random", "hot", "zero"):
        q.random_(-127, 128); rw.uniform_(); sa.uniform_(0, 0.001)
        if kind == "hot":
            ids.copy_(torch.randperm(256, device="cuda")[:6].expand_as(ids))
        elif kind == "zero":
            q.zero_()
        new = vendor.moe_align_block_size(ids, 128, 256, None)
        for dst, src in zip((si, ei, pad), new):
            dst.copy_(src)
        graph.replay()
        expected = torch.empty_like(out)
        vendor.moe_gemm_marlin_w8a8(q, w, expected, sa, sb, rw, si, ei, pad, 1, cfg)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        ri = torch.arange(24, device="cuda") * (m * 6 // 24)
        ni = torch.arange(24, device="cuda") * 127
        experts = ids.flatten()[ri]
        a = q[ri].cpu().long(); b = weight[0][experts, ni].cpu().long()
        acc = (a * b).sum(-1).float()
        effective = sa.flatten()[ri].cpu() * rw.flatten()[ri].cpu()
        cpu = ((acc * effective) * sb[experts, ni, 0].cpu()).bfloat16()
        torch.testing.assert_close(out.reshape(-1, 4096)[ri, ni].cpu(), cpu, atol=0, rtol=0)


@pytest.mark.parametrize("m", [16, 24, 32, 48, 64, 256, 1024, 4095, 8193])
def test_non_target_sizes_fall_back(m, weight):
    args, _ = case(m, weight)
    assert not impl.supports_prefill_down(*args)


@pytest.mark.parametrize("change", ["topk", "mode", "delta", "weight_layout", "scale_dtype", "router_stride"])
def test_unsupported_abi(change, weight):
    args, _ = case(4096, weight)
    if change == "topk": args[9] = 6
    elif change == "mode": args[10]["MODE"] = 94
    elif change == "delta": args[10]["DELTA"] = 1
    elif change == "weight_layout": args[1] = weight[0]
    elif change == "scale_dtype": args[3] = args[3].bfloat16()
    else: args[5] = torch.empty(4096, 12, device="cuda")[:, ::2]
    assert not impl.supports_prefill_down(*args)


def test_disable_and_private_binding(monkeypatch):
    original = vendor.fused_experts_impl_int8_marlin
    old_gemm = original.__globals__["moe_gemm_marlin_w8a8"]
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_MOE_GEMM2", "0")
    assert impl.adapt_prefill_moe_gemm2(original) is original
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_MOE_GEMM2", "1")
    adapted = impl.adapt_prefill_moe_gemm2(original)
    assert adapted is not original
    assert original.__globals__["moe_gemm_marlin_w8a8"] is old_gemm
    assert adapted.__globals__["moe_gemm_marlin_w8a8"] is not old_gemm


def test_mixed_lengths_reuse_compilation(weight):
    args, _ = case(8192, weight)
    impl.prefill_down(*args[:9])
    cache = impl._prefill_down.device_caches[torch.cuda.current_device()][0]
    before = len(cache)
    for m in range(8192, 8128, -1):
        q, w, out, sa, sb, rw, si, ei, pad = args[:9]
        # Keep a superset route plan; routes beyond this view are masked.
        impl.prefill_down(q[:m * 6], w, out[:m], sa[:m * 6], sb, rw[:m], si, ei, pad)
    torch.cuda.synchronize()
    assert len(cache) == before
