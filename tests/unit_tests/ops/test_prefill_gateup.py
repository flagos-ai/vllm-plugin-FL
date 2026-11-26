"""Narrow guards and small INT8 gate/up CPU-oracle coverage."""

import pytest
import torch

from vllm_fl.ops import prefill_gateup as gateup_op
from vllm_fl.patches import prefill_gateup as patch


def _cpu_round_away(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _cpu_gateup_int8(x, weight, scale):
    x_fp = x.float()
    sx = x_fp.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    qx = _cpu_round_away(x_fp / sx[:, None]).clamp(-127, 127).to(torch.int8)
    w_fp = weight.float()
    qw = torch.round(w_fp / scale[:, None]).clamp(-127, 127).to(torch.int8)
    accum = qx.to(torch.int32) @ qw.to(torch.int32).t()
    projection = (
        (accum.float() * sx[:, None] * scale[None, :]).to(torch.bfloat16).float()
    )
    h = weight.shape[0] // 2
    result = (torch.nn.functional.silu(projection[:, :h]) * projection[:, h:]).to(
        torch.bfloat16
    )
    return qx, sx, qw, result


def _cpu_gateup_prequantized_rowmajor(x, qweight, scale):
    x_fp = x.float()
    sx = x_fp.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    qx = _cpu_round_away(x_fp / sx[:, None]).clamp(-127, 127).to(torch.int8)
    accum = qx.to(torch.int32) @ qweight.to(torch.int32).t()
    projection = (
        (accum.float() * sx[:, None] * scale[None, :]).to(torch.bfloat16).float()
    )
    h = qweight.shape[0] // 2
    result = (torch.nn.functional.silu(projection[:, :h]) * projection[:, h:]).to(
        torch.bfloat16
    )
    return result


def _nrmse(actual, reference):
    diff = actual.float() - reference.float()
    return (
        diff.square().mean().sqrt() / reference.float().square().mean().sqrt()
    ).item()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA gate/up oracle",
)
@pytest.mark.gpu
@pytest.mark.parametrize("layout", ("packed", "rowmajor"))
def test_small_cuda_gateup_matches_independent_cpu_oracle(layout):
    m = 17
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(991 + m)
    k, n = 32, 64
    backing = torch.randn(
        (m, k + 3), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    x = backing[:, :k]
    weight = (
        torch.randn((n, k), device=device, dtype=torch.float32, generator=generator)
        .to(torch.bfloat16)
        .contiguous()
    )
    scale = gateup_op.weight_scales(weight)
    if m == 17:
        patch._cpu_numerical_selfcheck(weight, scale, layout=layout)

    qx, sx = gateup_op.quantize_activation_padded(x)
    if layout == "rowmajor":
        qw = gateup_op.quantize_weight_rowmajor(weight, scale)
        actual = gateup_op.prefill_gateup_int8_rowmajor(x, weight, scale).cpu()
    else:
        qw = gateup_op.pack_weight_transpose(weight, scale)
        actual = gateup_op.prefill_gateup_int8(x, weight, scale).cpu()

    x_cpu, weight_cpu, scale_cpu = x.cpu(), weight.cpu(), scale.cpu()
    qx_cpu, sx_cpu, qw_cpu, expected = _cpu_gateup_int8(x_cpu, weight_cpu, scale_cpu)
    assert torch.equal(qx[:m].cpu(), qx_cpu)
    torch.testing.assert_close(sx[:m].cpu(), sx_cpu, rtol=1e-6, atol=0.0)
    assert torch.count_nonzero(qx[m:]).item() == 0
    assert torch.count_nonzero(sx[m:]).item() == 0
    expected_qw = qw_cpu if layout == "rowmajor" else qw_cpu.t().contiguous()
    assert torch.equal(qw.cpu(), expected_qw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)

    fp_projection = x_cpu.float() @ weight_cpu.float().t()
    fp_projection = fp_projection.to(torch.bfloat16).float()
    h = n // 2
    fp_reference = (
        torch.nn.functional.silu(fp_projection[:, :h]) * fp_projection[:, h:]
    ).to(torch.bfloat16)
    assert _nrmse(actual, fp_reference) <= patch.MAX_NRMSE


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA prequantized row-major gate/up oracle",
)
@pytest.mark.gpu
@pytest.mark.parametrize("m", (1, 17))
def test_prequantized_rowmajor_matches_cpu_oracle_without_requantizing(monkeypatch, m):
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(1927)
    k, n = 32, 64
    backing = torch.randn(
        (m, k + 5), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    x = backing[:, :k]
    assert x.stride() == (k + 5, 1)
    source_weight = torch.randn(
        (n, k), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    scale = gateup_op.weight_scales(source_weight)
    qweight = torch.round(source_weight.float() / scale[:, None])
    qweight = qweight.clamp(-127, 127).to(torch.int8).contiguous()
    qweight_before = qweight.clone()
    on_demand = (
        gateup_op.prefill_gateup_int8_rowmajor(x, source_weight, scale)
        if m == 17
        else None
    )

    def unexpected_weight_quantization(*_args, **_kwargs):
        pytest.fail("prequantized candidate must not quantize BF16 weights")

    monkeypatch.setattr(
        gateup_op, "quantize_weight_rowmajor", unexpected_weight_quantization
    )
    monkeypatch.setattr(
        gateup_op, "pack_weight_transpose", unexpected_weight_quantization
    )
    actual = gateup_op.prefill_gateup_int8_prequantized_rowmajor(
        x, qweight, scale
    ).cpu()
    if on_demand is not None:
        assert torch.equal(actual, on_demand.cpu())
    expected = _cpu_gateup_prequantized_rowmajor(x.cpu(), qweight.cpu(), scale.cpu())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)
    assert torch.equal(qweight, qweight_before)

    # CUDA graph capture rejects a Python-side device readback in this call.
    gateup_op.prefill_gateup_int8_prequantized_rowmajor(x, qweight, scale)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gateup_op.prefill_gateup_int8_prequantized_rowmajor(
            x, qweight, scale
        )
    graph.replay()
    torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0.015625)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA opaque-op dynamic-shape regression",
)
@pytest.mark.parametrize("layout", ("packed", "rowmajor"))
@pytest.mark.gpu
def test_compiled_opaque_op_routes_small_m_to_bf16(monkeypatch, layout):
    device = torch.device("cuda", torch.cuda.current_device())
    from vllm.model_executor.layers.activation import SiluAndMul

    generator = torch.Generator(device=device).manual_seed(4771)
    k, n, threshold = 32, 64, patch.MIN_PREFILL_TOKENS
    weight = (
        torch.randn((n, k), device=device, dtype=torch.float32, generator=generator)
        .to(torch.bfloat16)
        .contiguous()
    )
    scale = gateup_op.weight_scales(weight)
    native_activation = torch.compile(SiluAndMul.forward_native, dynamic=True)
    monkeypatch.setattr(gateup_op, "_small_m_activation", native_activation)
    calls = 0
    quantized_impl = (
        gateup_op.prefill_gateup_int8_rowmajor
        if layout == "rowmajor"
        else gateup_op.prefill_gateup_int8
    )

    def count_quantized_calls(x, w, s):
        nonlocal calls
        calls += 1
        return quantized_impl(x, w, s)

    quantized_name = (
        "prefill_gateup_int8_rowmajor"
        if layout == "rowmajor"
        else "prefill_gateup_int8"
    )
    monkeypatch.setattr(gateup_op, quantized_name, count_quantized_calls)
    torch._dynamo.reset()
    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    opaque_op = (
        torch.ops.vllm_fl.prefill_gateup_int8_rowmajor
        if layout == "rowmajor"
        else torch.ops.vllm_fl.prefill_gateup_int8
    )
    compiled = torch.compile(
        lambda x: opaque_op(x, weight, scale, threshold),
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )
    large = torch.randn(
        (threshold, k), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    assert compiled(large).shape == (threshold, n // 2)
    assert calls == 1
    assert compile_count == 1

    for m in (16, 64, 1024):
        x = torch.randn(
            (m, k), device=device, dtype=torch.float32, generator=generator
        ).to(torch.bfloat16)
        actual = compiled(x)
        projection = torch.mm(x, weight.t())
        expected = native_activation(projection)
        assert torch.equal(actual, expected)
        assert calls == 1
        assert compile_count == 1
    x = torch.randn((1, k), device=device, dtype=torch.float32, generator=generator).to(
        torch.bfloat16
    )
    actual = opaque_op(x, weight, scale, threshold)
    expected = native_activation(torch.mm(x, weight.t()))
    assert torch.equal(actual, expected)
    assert calls == 1
    empty = torch.empty((0, k), device=device, dtype=torch.bfloat16)
    assert opaque_op(empty, weight, scale, threshold).shape == (0, n // 2)
    assert calls == 1
    torch._dynamo.reset()
