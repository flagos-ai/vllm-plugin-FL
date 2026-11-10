# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon hipBLASLt bf16->fp32 GEMM (compressor projections) unit tests.

Run with::

    pytest -q -s tests/unit_tests/dispatch/test_hygon_bf16_fp32_gemm.py
"""

from __future__ import annotations

import pytest
import torch

# (名称, N, M) —— 覆盖三条路由分支: N>=1024 / N<1024&M<=1024 / N<1024&M>1024
CASES = [
    ("comp_r128", 1024, 64),    # 分支1: hipBLASLt 默认 algo
    ("comp_r4", 2048, 36),      # 分支1
    ("idx_comp", 512, 64),      # 分支2: hipBLASLt heuristic
    ("weights_proj", 64, 24),   # 分支2
    ("idx_comp_prefill", 512, 4096),   # 分支3: 回落 torch.mm (hipBLASLt 坏区间)
]

K = 4096


@pytest.mark.parametrize("n", [512, 1024, 2048])
@pytest.mark.parametrize("m", [1, 8, 31, 64, 65, 128, 129, 192, 193, 255, 256])
def test_decode_projection_graph_replay(gemm_mod, n, m):
    torch.manual_seed(915)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(n, K, device="cuda", dtype=torch.bfloat16) * 0.05
    gemm_mod.linear_bf16_fp32_hygon(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gemm_mod.linear_bf16_fp32_hygon(x, w)
    for _ in range(3):
        x.normal_(std=0.5)
        w.normal_(std=0.05)
        graph.replay()
        expected = x.float() @ w.float().T
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
        first = out.clone()
        graph.replay()
        assert torch.equal(first, out)


def test_all_decode_batch_sizes_select_splitk(gemm_mod, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.gemm import decode_projection

    sentinel = object()
    calls = []

    def selected(x, w):
        calls.append((x.shape[0], w.shape[0]))
        return sentinel

    monkeypatch.setenv("VLLM_FL_BF16_FP32_GEMM", "blaslt")
    monkeypatch.setenv("VLLM_FL_HYGON_DECODE_SPLITK_GEMM", "1")
    monkeypatch.setattr(decode_projection, "decode_projection_bf16_fp32", selected)
    x = torch.empty(256, K, device="cuda", dtype=torch.bfloat16)
    for n in (512, 1024, 2048):
        w = torch.empty(n, K, device="cuda", dtype=torch.bfloat16)
        for m in range(1, 257):
            assert gemm_mod.linear_bf16_fp32_hygon(x[:m], w) is sentinel
    assert len(calls) == 256 * 3


@pytest.fixture(scope="module")
def gemm_mod():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Hygon/HIP GPU is required")
    from vllm_fl.dispatch.backends.vendor.hygon.impl.gemm import (
        bf16_fp32_gemm as mod,
    )

    return mod


def _assert_close(name, actual, expected):
    ref = expected.float()
    diff = (actual.float() - ref).abs()
    rel = (diff / (ref.abs() + 1e-3)).mean().item()
    print(f"{name}: max_abs={diff.max().item():.6f} mean_rel={rel:.2e}")
    assert not torch.isnan(actual).any() and not torch.isinf(actual).any()
    assert rel < 2e-2, f"{name}: mean_rel={rel}"


@pytest.mark.parametrize("name,N,M", CASES)
def test_linear_matches_torch_mm(gemm_mod, name, N, M):
    torch.manual_seed(hash(name) % 10000)
    x = (torch.randn((M, K), device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn((N, K), device="cuda") * 0.05).to(torch.bfloat16)
    expected = torch.mm(x, w.T, out_dtype=torch.float32)
    _assert_close(f"{name}({M}x{N}x{K})", gemm_mod.linear_bf16_fp32_hygon(x, w), expected)


def test_fallbacks_to_torch_for_non_bf16(gemm_mod):
    x = torch.randn(64, K, device="cuda", dtype=torch.float32)
    w = torch.randn(1024, K, device="cuda", dtype=torch.float32)
    out = gemm_mod.linear_bf16_fp32_hygon(x, w)
    expected = torch.mm(x, w.T, out_dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_env_switch_restores_torch(gemm_mod, monkeypatch):
    monkeypatch.setenv("VLLM_FL_BF16_FP32_GEMM", "torch")
    gemm_mod._blaslt_available.cache_clear()
    try:
        assert not gemm_mod._blaslt_available()
        torch.manual_seed(7)
        x = (torch.randn(64, K, device="cuda") * 0.5).to(torch.bfloat16)
        w = (torch.randn(1024, K, device="cuda") * 0.05).to(torch.bfloat16)
        out = gemm_mod.linear_bf16_fp32_hygon(x, w)
        expected = torch.mm(x, w.T, out_dtype=torch.float32)
        assert torch.equal(out, expected), "switch= torch 必须逐位走 torch.mm"
    finally:
        monkeypatch.delenv("VLLM_FL_BF16_FP32_GEMM", raising=False)
        gemm_mod._blaslt_available.cache_clear()


def test_attention_method_patched(gemm_mod):
    from vllm_fl.dispatch.backends.vendor.hygon.patches.models import deepseek_v4 as patches

    patches.patch_deepseek_v4_bf16_fp32_gemm()

    import vllm.models.deepseek_v4.attention as attention_mod

    assert getattr(
        attention_mod.DeepseekV4Attention.attn_gemm_parallel_execute,
        "_vllm_fl_hygon",
        False,
    ), "attn_gemm_parallel_execute 应被 hygon patch 标记"
    # 幂等
    patches.patch_deepseek_v4_bf16_fp32_gemm()


def test_patched_method_end_to_end(gemm_mod):
    """伪 layer 走 patched 方法: 无 compressor/indexer 时与原逻辑等价返回."""
    import vllm.models.deepseek_v4.attention as attention_mod
    from vllm_fl.dispatch.backends.vendor.hygon.patches.models import deepseek_v4 as patches

    patches.patch_deepseek_v4_bf16_fp32_gemm()
    torch.manual_seed(11)
    hidden = (torch.randn(8, 128, device="cuda") * 0.5).to(torch.bfloat16)

    w = torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda")

    class _Layer:
        aux_stream_list = None
        compressor = None
        indexer = None
        ln_events = [torch.cuda.Event() for _ in range(4)]

        def fused_wqa_wkv(self, h):
            return w(h), None  # vllm Linear 返回 (output, bias)

    out = attention_mod.DeepseekV4Attention.attn_gemm_parallel_execute(_Layer(), hidden)
    qr_kv, kv_score, indexer_kv_score, indexer_weights = out
    assert kv_score is None and indexer_kv_score is None and indexer_weights is None
    assert torch.allclose(qr_kv, w(hidden)), "fused_wqa_wkv 分支必须保持原行为"
