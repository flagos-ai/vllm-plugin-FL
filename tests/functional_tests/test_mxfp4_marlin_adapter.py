# SPDX-License-Identifier: Apache-2.0
"""Hopper numerical validation for the MXFP4 service adapter.

This is a correctness test, not a replacement throughput benchmark.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config

from vllm_fl.ops.fused_moe.mxfp4_marlin import _make_experts_class
from vllm_fl.ops.fused_moe.raw_retirement import assert_retired_aliases


def _dequant(w, s):
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    sign = torch.where(codes & 8 != 0, -1.0, 1.0)
    return (
        table[(codes & 7).long()]
        * sign
        * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA Hopper")
@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("shape", [(128, 128, 2), (4096, 256, 6)])
@pytest.mark.parametrize("weight_input", [False, True])
def test_hopper_split_mxfp4_matches_native_activation(m, shape, weight_input):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlagGems MXFP4 is currently limited to Hopper")
    torch.manual_seed(311)
    h, i, topk = shape
    e = 8
    device, dtype = "cuda", torch.bfloat16
    w1 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device=device)
    s1 = torch.randint(118, 123, (e, 2 * i, h // 32), dtype=torch.uint8, device=device)
    s2 = torch.randint(118, 123, (e, h, i // 32), dtype=torch.uint8, device=device)
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
    quant = mxfp4_w4a16_moe_quant_config(
        w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0
    )
    experts = _make_experts_class()(moe_config=moe, quant_config=quant)
    dense1, dense2 = _dequant(w1, s1), _dequant(w2, s2)
    raw_copies = [p.clone() for p in (w1, w2, s1, s2)]
    raw_storage_refs = [p.untyped_storage()._weak_ref() for p in (w1, w2, s1, s2)]
    experts.process_weights_after_loading(layer)
    from vllm_fl.ops.fused_moe.mxfp4_marlin_invoke import _invoke_w4a16_mxfp4_moe_gemm

    assert experts.gemm1 is _invoke_w4a16_mxfp4_moe_gemm
    assert all(torch.UntypedStorage._expired(r) for r in raw_storage_refs)
    for ref in raw_storage_refs:
        torch.UntypedStorage._free_weak_ref(ref)
    pointers = [p.data_ptr() for p in experts.packed]
    original_pack = experts.gems.w4a16_mxfp4_pack

    def refuse_repack(*args, **kwargs):
        raise AssertionError("second processing attempted to repack retired bytes")

    experts.gems.w4a16_mxfp4_pack = refuse_repack
    try:
        experts.process_weights_after_loading(layer)
    finally:
        experts.gems.w4a16_mxfp4_pack = original_pack
    assert pointers == [p.data_ptr() for p in experts.packed]
    assert_retired_aliases(layer, experts.packed)
    x = (torch.randn(m, h, device=device) * 10).to(dtype)
    ids = torch.stack([torch.randperm(e, device=device)[:topk] for _ in range(m)])
    weights = torch.softmax(torch.randn(m, topk, device=device), -1)
    output = torch.empty_like(x)
    experts.apply(
        output,
        x,
        w1,
        w2,
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
    reference = torch.empty_like(x)
    reference_routes = torch.empty((m, topk, h), device=device, dtype=dtype)
    for t in range(m):
        routes = []
        for r in range(topk):
            expert = ids[t, r]
            first = dense1[expert] @ x[t].float()
            if weight_input:
                first *= weights[t, r]
            middle = torch.empty((1, i), device=device, dtype=dtype)
            experts.activation(MoEActivation.SILU, middle, first.to(dtype)[None])
            second = dense2[expert] @ middle[0].float()
            if not weight_input:
                second *= weights[t, r]
            routes.append(second.to(dtype))
        reference_routes[t] = torch.stack(routes)
        reference[t] = reference_routes[t].sum(0)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
    # The first warmup must have populated packs; a captured replay cannot repack
    # weights or read E8M0 bounds back to the CPU.
    graph = torch.cuda.CUDAGraph()
    for _ in range(2):
        experts.apply(
            output,
            x,
            w1,
            w2,
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
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        experts.apply(
            output,
            x,
            w1,
            w2,
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
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
    # Exercise canonical EP maps with non-contiguous global expert placement.
    # Actual multi-node transport still needs a real service run.
    partials = []
    partial_references = []
    partial_error_bounds = []
    for rank in range(2):
        owned = torch.arange(rank, e, 2, device=device)
        mapping = torch.full((e,), -1, dtype=torch.int32, device=device)
        mapping[owned] = torch.arange(e // 2, dtype=torch.int32, device=device)
        local_layer = SimpleNamespace(
            w13_weight=raw_copies[0][owned].contiguous(),
            w2_weight=raw_copies[1][owned].contiguous(),
        )
        local_moe = SimpleNamespace(**vars(moe))
        local_moe.num_local_experts = e // 2
        local_layer.w13_weight_scale = raw_copies[2][owned].contiguous()
        local_layer.w2_weight_scale = raw_copies[3][owned].contiguous()
        local_quant = mxfp4_w4a16_moe_quant_config(
            w1_scale=local_layer.w13_weight_scale,
            w2_scale=local_layer.w2_weight_scale,
            gemm1_clamp_limit=10.0,
        )
        local_experts = _make_experts_class()(
            moe_config=local_moe, quant_config=local_quant
        )
        local_experts.process_weights_after_loading(local_layer)
        partial = torch.empty_like(x)
        captured = {}
        original_gemm = local_experts.gemm1

        def trace_gemm(_gemm=original_gemm, _captured=captured, **kw):
            _gemm(**kw)
            if kw["C"].ndim == 3:
                _captured["routes"] = kw["C"].clone()

        local_experts.gemm1 = trace_gemm
        local_experts.apply(
            partial,
            x,
            local_layer.w13_weight,
            local_layer.w2_weight,
            weights,
            ids,
            MoEActivation.SILU,
            e,
            mapping,
            None,
            None,
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            None,
            weight_input,
        )
        local_experts.gemm1 = original_gemm
        partials.append(partial)
        owned_mask = (ids.remainder(2) == rank).unsqueeze(-1)
        local_reference = reference_routes.masked_fill(~owned_mask, 0).sum(1)
        partial_references.append(local_reference)
        local_delta = (partial.float() - local_reference.float()).abs()
        print(
            f"EP rank={rank} M={m} H={h} input_weight={weight_input} "
            f"local_max={local_delta.max().item():.8g} local_rmse="
            f"{local_delta.square().mean().sqrt().item():.8g}"
        )
        actual_routes = captured["routes"].masked_fill(~owned_mask, 0)
        route_delta = (
            actual_routes.float() - reference_routes.masked_fill(~owned_mask, 0).float()
        ).abs()
        print(f"route_max={route_delta.max().item():.8g}")
        expected_routes = reference_routes.masked_fill(~owned_mask, 0)
        # Compare BEFORE reduction, where cancellation cannot turn one BF16 ULP
        # on a ~20-valued expert into a 20% error on its near-zero routed sum.
        # 0.8% is approximately one BF16 ULP, stricter than the 3% full-output
        # comparison above. Keep an absolute allowance for near-zero GEMMs.
        torch.testing.assert_close(
            actual_routes, expected_routes, rtol=0.008, atol=0.03
        )
        route_sum = actual_routes.float().sum(1)
        reduce_bound = torch.finfo(dtype).eps * route_sum.abs() + topk * torch.finfo(
            torch.float32
        ).eps * actual_routes.float().abs().sum(1)
        assert torch.all((partial.float() - route_sum).abs() <= reduce_bound)
        eps = torch.finfo(dtype).eps
        # Triangle inequality propagates the SAME per-route tolerances through
        # rank-local BF16 summation; do not add a new arbitrary final tolerance.
        error_bound = (
            0.008 * expected_routes.float().abs().sum(1)
            + 0.03 * owned_mask.sum(1)
            + eps * local_reference.float().abs()
        )
        partial_error_bounds.append(error_bound)
        assert torch.all(local_delta <= error_bound), (
            "rank sum exceeds BF16 route error bound"
        )
    expected_combined = sum(partial_references)
    actual_combined = sum(partials)
    old_reference_delta = (expected_combined.float() - reference.float()).abs()
    delta = (actual_combined.float() - expected_combined.float()).abs()
    print(
        f"EP combined M={m} H={h} input_weight={weight_input} "
        f"two_stage_rounding_max={old_reference_delta.max().item():.8g} "
        f"corrected_max={delta.max().item():.8g} corrected_rmse="
        f"{delta.square().mean().sqrt().item():.8g}"
    )
    combined_bound = sum(partial_error_bounds) + torch.finfo(dtype).eps * sum(
        p.float().abs() for p in partial_references
    )
    assert torch.all(delta <= combined_bound), (
        "combined EP exceeds propagated BF16 error bound"
    )
    normalized_rmse = (
        delta.square().mean().sqrt() / expected_combined.float().square().mean().sqrt()
    )
    print(f"EP combined normalized_rmse={normalized_rmse.item():.8g}")
    assert normalized_rmse < 0.003

    x.mul_(0.5)
    changed_expected = torch.empty_like(output)
    experts.apply(
        changed_expected,
        x,
        w1,
        w2,
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
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, changed_expected, rtol=0, atol=0)

    w1.data = w1.clone()
    with pytest.raises(RuntimeError, match="moved/changed"):
        experts.apply(
            output,
            x,
            w1,
            w2,
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
