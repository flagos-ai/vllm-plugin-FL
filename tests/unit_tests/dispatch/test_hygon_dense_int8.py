# Copyright (c) 2026 BAAI. All rights reserved.
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.impl.gemm import dense_int8 as impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Hygon GPU required",
)


@pytest.mark.parametrize("m", [1, 31, 64, 65, 128, 256])
@pytest.mark.parametrize("n,k", [(1536, 4096), (512, 4096), (4096, 1024)])
def test_changed_input_graph(m, n, k):
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).T
    sa = torch.rand(m, 1, device="cuda")
    sb = torch.rand(n, 1, device="cuda")
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    args = (a, b, sa, sb, torch.bfloat16, bias)
    impl.hygon_triton_scaled_mm(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = impl.hygon_triton_scaled_mm(*args)
    for value in [-128, 0, 127]:
        a.fill_(value)
        graph.replay()
        expected = impl._original_scaled_mm(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("case", [
    "disabled", "large_m", "empty_m", "cpu", "dtype", "output_dtype",
    "weight_layout", "input_layout", "shape", "scale_layout", "scale_shape",
    "scale_dtype", "bias_dtype", "explicit_tiles", "no_heuristic",
])
def test_fallback_preserves_arguments(case, monkeypatch):
    a = torch.empty(64, 4096, device="cuda", dtype=torch.int8)
    b = torch.empty(512, 4096, device="cuda", dtype=torch.int8).T
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(512, 1, device="cuda")
    out_dtype, bias, kw = torch.bfloat16, None, {}
    if case == "disabled":
        monkeypatch.setenv("VLLM_FL_HYGON_DENSE_INT8_TUNING", "0")
    elif case in {"large_m", "empty_m"}:
        a = torch.empty(257 if case == "large_m" else 0, 4096, device="cuda", dtype=a.dtype)
    elif case == "cpu":
        a = a.cpu()
    elif case == "dtype":
        a, b = a.float(), b.float()
    elif case == "output_dtype":
        out_dtype = torch.float32
    elif case == "weight_layout":
        b = b.contiguous()
    elif case == "input_layout":
        a = torch.empty(64, 8192, device="cuda", dtype=a.dtype)[:, ::2]
    elif case == "shape":
        b = b[:, :256]
    elif case == "scale_layout":
        sa = torch.ones(128, 1, device="cuda")[::2]
    elif case == "scale_shape":
        sa = sa.reshape(1, 64)
    elif case == "scale_dtype":
        sa = sa.bfloat16()
    elif case == "bias_dtype":
        bias = torch.ones(512, device="cuda")
    elif case == "explicit_tiles":
        kw["block_size_m"] = 16
    elif case == "no_heuristic":
        kw["use_heuristic"] = False
    sentinel, calls = object(), []

    def fallback(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(impl, "_original_scaled_mm", fallback)
    assert impl.hygon_triton_scaled_mm(a, b, sa, sb, out_dtype, bias, **kw) is sentinel
    assert len(calls) == 1
    assert all(x is y for x, y in zip(calls[0][:6], (a, b, sa, sb, out_dtype, bias)))
    assert calls[0][6:] == (kw.get("block_size_m", 32), 32, 32, kw.get("use_heuristic", True))
