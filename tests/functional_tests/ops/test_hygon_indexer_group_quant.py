"""Minimal numerical/layout/fake/Graph checks for the Hygon DSA quant bridge."""
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.hygon.patches.models.glm_group_quant import (
    per_token_group_quant_fp8,
)


@pytest.mark.parametrize("column,tma", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_values_layout_and_out(column, tma, ue8m0):
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    x_cpu = torch.linspace(-3, 3, 5 * 256).reshape(5, 256).bfloat16()
    x_cpu[0].zero_()
    x_cpu[1].mul_(1e-12)
    x = x_cpu.cuda()
    dtype = fp8_utils.current_platform.fp8_dtype()
    out = torch.empty_like(x, dtype=dtype)
    q, s = per_token_group_quant_fp8(
        x, 128, column_major_scales=column, tma_aligned_scales=tma,
        out_q=out, use_ue8m0=ue8m0)
    lo, hi = fp8_utils.get_fp8_min_max()
    grouped = x_cpu.float().reshape(5, 2, 128)
    expected_s = grouped.abs().amax(-1).clamp_min(1e-10) * (1.0 / hi)
    if ue8m0:
        expected_s = torch.exp2(torch.ceil(torch.log2(expected_s)))
    expected_q = (grouped / expected_s[..., None]).clamp(lo, hi).to(dtype)
    assert q.data_ptr() == out.data_ptr()
    torch.testing.assert_close(s.cpu(), expected_s, rtol=1e-6, atol=0)
    torch.testing.assert_close(q.float().cpu().reshape(5, 2, 128),
                               expected_q.float(), rtol=0, atol=0)
    assert s.stride() == ((1, 8 if tma else 5) if column else (2, 1))


def test_compile_and_graph_replay():
    def quant(x):
        q, s = per_token_group_quant_fp8(x, 128, use_ue8m0=True,
                                        column_major_scales=True,
                                        tma_aligned_scales=True)
        return q, s
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(quant, backend="eager", fullgraph=True)
    expected = quant(x)
    actual = compiled(x)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            quant(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_q, replay_s = quant(x)
    x.mul_(0.25)
    expected = quant(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_q.float(), expected[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(replay_s, expected[1], rtol=0, atol=0)


def test_invalid_shape():
    with pytest.raises(ValueError, match="group size"):
        per_token_group_quant_fp8(torch.empty(5, 129, device="cuda"), 128)
