# SPDX-License-Identifier: Apache-2.0
"""New K64 logic: dense reference, original padded K128, graph and tail groups."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config

from vllm_fl.ops.fused_moe.mxfp4_marlin import (
    _make_experts_class as compact_class,
    _make_experts_class as original_class,
)
from vllm_fl.ops.fused_moe.raw_retirement import assert_retired_aliases


def dequant(w, s):
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    return (
        table[(codes & 7).long()]
        * torch.where(codes & 8 != 0, -1.0, 1.0)
        * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
    )


@pytest.mark.parametrize("m", [1, 16, 128])
@pytest.mark.parametrize("weight_input", [False, True])
@pytest.mark.parametrize("fold", [False, True])
def test_unpadded_i192_clamp_scale_groups_and_graph(m, weight_input, fold, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")
    torch.manual_seed(195)
    e, h, original_i, compact_i, topk = 8, 7168, 256, 192, 6
    device, dtype = "cuda", torch.bfloat16
    w1 = torch.zeros((e, original_i * 2, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.zeros((e, h, original_i // 2), dtype=torch.uint8, device=device)
    w1[:, :compact_i] = torch.randint(
        0, 256, (e, compact_i, h // 2), dtype=torch.uint8, device=device
    )
    w1[:, original_i : original_i + compact_i] = torch.randint(
        0, 256, (e, compact_i, h // 2), dtype=torch.uint8, device=device
    )
    w2[:, :, : compact_i // 2] = torch.randint(
        0, 256, (e, h, compact_i // 2), dtype=torch.uint8, device=device
    )
    s1 = torch.randint(
        118, 124, (e, original_i * 2, h // 32), dtype=torch.uint8, device=device
    )
    s2 = torch.empty((e, h, original_i // 32), dtype=torch.uint8, device=device)
    # Distinct non-periodic valid group scales expose wrong s0/s0/s1/s1 indexing.
    valid_scales = torch.tensor(
        [118, 123, 120, 124, 119, 122], dtype=torch.uint8, device=device
    )
    s2[:, :, : compact_i // 32] = valid_scales
    s2[:, :, compact_i // 32 :] = 200  # Poison UNUSED loader tail scales.
    if not fold:
        # A live exponent129 disables BF16 fold; activation remains BF16.
        # Its FP4 weight code is zero to keep outputs finite and controlled.
        s1[:, 0, 0] = 129
        w1[:, 0, :16] = 0
    dense1, dense2 = dequant(w1, s1), dequant(w2, s2)
    reference_layer = SimpleNamespace(w13_weight=w1.clone(), w2_weight=w2.clone())
    reference_quant = mxfp4_w4a16_moe_quant_config(
        w1_scale=s1.clone(), w2_scale=s2.clone(), gemm1_clamp_limit=10.0
    )
    layer = SimpleNamespace(
        w13_weight=w1, w2_weight=w2, w13_weight_scale=s1, w2_weight_scale=s2
    )
    moe = SimpleNamespace(
        hidden_dim=h,
        in_dtype=dtype,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        num_experts=e,
        num_local_experts=e,
        activation=MoEActivation.SILU,
    )
    baseline = original_class()(moe_config=moe, quant_config=reference_quant)
    # The baseline always uses the unchanged original K128 layout/kernel,
    # regardless of whether this source checkout also contains retirement.
    with monkeypatch.context() as baseline_env:
        baseline_env.setenv("VLLM_FL_MXFP4_RETIRE_RAW", "0")
        baseline_env.setenv("VLLM_FL_MXFP4_COMPACT_I", "0")
        baseline.process_weights_after_loading(reference_layer)
    monkeypatch.setenv("VLLM_FL_MXFP4_COMPACT_I", "192")
    quant = mxfp4_w4a16_moe_quant_config(
        w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0
    )
    experts = compact_class()(moe_config=moe, quant_config=quant)
    raw_refs = [p.untyped_storage()._weak_ref() for p in (w1, w2, s1, s2)]
    experts.process_weights_after_loading(layer)
    assert experts.fold_scale is fold
    assert experts.second_block_k == 64
    assert layer.w13_weight.shape[1] == compact_i * 2
    assert layer.w2_weight.shape[-1] == compact_i // 2
    assert all(torch.UntypedStorage._expired(r) for r in raw_refs)
    for r in raw_refs:
        torch.UntypedStorage._free_weak_ref(r)
    assert_retired_aliases(layer, experts.packed)
    pointers = [p.data_ptr() for p in experts.packed]
    experts.process_weights_after_loading(layer)
    assert pointers == [p.data_ptr() for p in experts.packed]
    x = (torch.randn(m, h, device=device) * 10).to(dtype)
    ids = torch.stack([torch.randperm(e, device=device)[:topk] for _ in range(m)])
    weights = torch.softmax(torch.randn(m, topk, device=device), -1)
    out, native = torch.empty_like(x), torch.empty_like(x)

    def apply(impl, target, loaded):
        impl.apply(
            target,
            x,
            loaded.w13_weight,
            loaded.w2_weight,
            weights,
            ids,
            MoEActivation.SILU,
            e,
            None,
            None,
            None,
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            None,
            weight_input,
        )

    captured = {}
    original_native_gemm = baseline.gemm1
    original_first_gemm = experts.gemm1
    original_k64_gemm = experts.gemm2

    def trace_native(**kw):
        original_native_gemm(**kw)
        if kw["C"].ndim == 3:
            captured["native"] = kw["C"].clone()
        else:
            captured["native_first"] = kw["C"].clone()

    def trace_first(**kw):
        original_first_gemm(**kw)
        if kw["C"].ndim == 2:
            captured["first"] = kw["C"].clone()

    def trace_k64(**kw):
        original_k64_gemm(**kw)
        captured["k64"] = kw["C"].clone()

    experts.gemm1 = trace_first
    experts.gemm2 = trace_k64
    try:
        apply(experts, out, layer)
        baseline.gemm1 = trace_native
        apply(baseline, native, reference_layer)
    finally:
        baseline.gemm1 = original_native_gemm
        experts.gemm1 = original_first_gemm
        experts.gemm2 = original_k64_gemm
    expected = torch.empty_like(x)
    reference_routes = torch.empty((m, topk, h), device=device, dtype=dtype)
    reference_first = torch.empty((m * topk, 2 * compact_i), device=device, dtype=dtype)
    full_reference = torch.empty_like(reference_routes)
    for t in range(m):
        routes = []
        for r in range(topk):
            first = dense1[ids[t, r]] @ x[t].float()
            if weight_input:
                first *= weights[t, r]
            reference_first[t * topk + r] = torch.cat(
                (first[:compact_i], first[original_i : original_i + compact_i])
            ).to(dtype)
            middle = torch.empty((1, original_i), device=device, dtype=dtype)
            baseline.activation(MoEActivation.SILU, middle, first.to(dtype)[None])
            second = dense2[ids[t, r]] @ middle[0].float()
            if not weight_input:
                second *= weights[t, r]
            full_reference[t, r] = second.to(dtype)
            # Check each GEMM independently. In a two-layer FP32 reference,
            # a GEMM1 BF16 rounding boundary can be amplified by SwiGLU;
            # do not attribute that inherited difference to the new K64 GEMM.
            actual_middle = torch.empty((1, compact_i), device=device, dtype=dtype)
            experts.activation(
                MoEActivation.SILU, actual_middle, captured["first"][t * topk + r][None]
            )
            second_stage_reference = (
                dense2[ids[t, r], :, :compact_i] @ actual_middle[0].float()
            )
            if not weight_input:
                second_stage_reference *= weights[t, r]
            routes.append(second_stage_reference.to(dtype))
        reference_routes[t] = torch.stack(routes)
        expected[t] = reference_routes[t].sum(0)
    torch.cuda.synchronize()
    print(
        "K64 native bitwise diagnostic max=",
        (out.float() - native.float()).abs().max().item(),
    )
    torch.testing.assert_close(captured["k64"], captured["native"], rtol=0, atol=0)
    torch.testing.assert_close(out, native, rtol=0, atol=0)
    torch.testing.assert_close(
        captured["first"], reference_first, rtol=0.008, atol=0.03
    )
    torch.testing.assert_close(
        captured["first"],
        torch.cat(
            (
                captured["native_first"][:, :compact_i],
                captured["native_first"][:, original_i : original_i + compact_i],
            ),
            1,
        ),
        rtol=0,
        atol=0,
    )
    # Independent reference is checked BEFORE cancellation, using ~one BF16 ULP
    # per expert. Propagate that same error bound through top-k BF16 reduction.
    torch.testing.assert_close(captured["k64"], reference_routes, rtol=0.008, atol=0.03)
    # Different FP32 reduction trees may round a near-cancellation result to
    # adjacent BF16 values. Bound FP32 accumulation plus one BF16 rounding;
    # new K64 vs original padded K128 remains BITWISE checked above.
    route_sum = captured["k64"].float().sum(1)
    reduce_bound = torch.finfo(dtype).eps * route_sum.abs() + topk * torch.finfo(
        torch.float32
    ).eps * captured["k64"].float().abs().sum(1)
    assert torch.all((out.float() - route_sum).abs() <= reduce_bound)
    error_bound = (
        0.008 * reference_routes.float().abs().sum(1)
        + 0.03 * topk
        + torch.finfo(dtype).eps * expected.float().abs()
    )
    delta = (out.float() - expected.float()).abs()
    assert torch.all(delta <= error_bound)
    normalized_rmse = (
        delta.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )
    assert normalized_rmse < 0.003
    full_sum = full_reference.float().sum(1)
    end_to_end_rmse = (
        out.float() - full_sum
    ).square().mean().sqrt() / full_sum.square().mean().sqrt()
    assert end_to_end_rmse < 0.003
    expected_graph = out.clone()
    for _ in range(2):
        apply(experts, out, layer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply(experts, out, layer)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected_graph, rtol=0, atol=0)
    x.mul_(0.5)
    changed_expected = torch.empty_like(out)
    apply(experts, changed_expected, layer)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, changed_expected, rtol=0, atol=0)
    print(
        f"K64 M={m} input_weight={weight_input} fold={fold} max_native="
        f"{(expected_graph.float() - native.float()).abs().max().item()}"
    )
