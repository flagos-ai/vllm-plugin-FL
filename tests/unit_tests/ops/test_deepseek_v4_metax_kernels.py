# SPDX-License-Identifier: Apache-2.0
"""Explicitly opt-in GPU regressions; never run as a CPU/registration check."""

import importlib
import os

import pytest

if os.getenv("DSV4_RUN_GPU_TESTS") != "1":
    pytest.skip(
        "Set DSV4_RUN_GPU_TESTS=1 on an isolated MetaX GPU", allow_module_level=True
    )

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

PREFIX = "vllm_fl.ops.deepseek_v4_metax"


@pytest.mark.parametrize("topk", [129, 640, 768])
@pytest.mark.parametrize("with_sink", [False, True])
def test_staged_prefill_preserves_serial_chunks(topk, with_sink):
    serial = importlib.import_module(PREFIX + ".prefill")
    staged = importlib.import_module(PREFIX + ".attention_staged")
    torch.manual_seed(1234)
    t, n, scale = 8, 1024, 0.037
    q = torch.randn(t, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(n, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(n, (t, 1, topk), device="cuda", dtype=torch.int32)
    idx[:, :, 7::19] = -1
    lens = torch.tensor(
        [0, 1, 15, 16, 17, 31, topk - 1, topk], device="cuda", dtype=torch.int32
    )
    sink = torch.randn(16, device="cuda", dtype=torch.float32) if with_sink else None
    expected = torch.empty_like(q)
    ml = torch.empty((t, 16), device="cuda", dtype=torch.float32)
    lse = torch.empty_like(ml)
    for start, stats in ((0, True), (256, False)):
        serial.triton_flash_mla_sparse_fwd_halfd_serial[(t,)](
            q,
            kv,
            idx,
            sink,
            lens,
            scale,
            expected,
            ml,
            lse,
            q.stride(1),
            q.stride(0),
            kv.stride(1),
            kv.stride(0),
            idx.stride(1),
            idx.stride(0),
            expected.stride(1),
            expected.stride(0),
            ml.stride(0),
            lse.stride(0),
            t,
            16,
            512,
            n,
            topk,
            sink is not None,
            True,
            BK=16,
            BH=16,
            V_START=start,
            WRITE_STATS=stats,
            num_warps=2,
            num_stages=1,
        )
    got, got_ml, got_lse = (
        torch.empty_like(q),
        torch.empty_like(ml),
        torch.empty_like(lse),
    )
    staged.run(
        q,
        kv,
        idx,
        sink,
        lens,
        got,
        got_ml,
        got_lse,
        staged.buffers(q, topk),
        sm_scale=scale,
    )
    for a, b in ((got, expected), (got_ml, ml), (got_lse, lse)):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=False)


@pytest.mark.parametrize("tokens", [17, 1024, 4032])
def test_mhc_tail_against_fused_reference(tokens):
    split = importlib.import_module(PREFIX + ".mhc_split")
    torch.manual_seed(1234)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(tokens, 24, device="cuda", dtype=torch.bfloat16).float()
    scale = torch.randn(3, device="cuda", dtype=torch.float32)
    bias = torch.randn(24, device="cuda", dtype=torch.float32)
    post = torch.empty((tokens, 4), device="cuda", dtype=torch.float32)
    comb = torch.empty((tokens, 16), device="cuda", dtype=torch.float32)
    layer = torch.empty((tokens, 4096), device="cuda", dtype=torch.bfloat16)
    split._original(
        g, scale, bias, residual, post, comb, layer, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    actual = split.tail(g, scale, bias, residual, 1e-6, 1e-6, 1e-6, 2.0, 20)
    for a, b in zip(actual, (post, comb, layer)):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=False)
