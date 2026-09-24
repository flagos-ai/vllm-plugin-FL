# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.common import hyperconnection as hc


def _config() -> hc.HyperConnectionConfig:
    return hc.HyperConnectionConfig(
        hc_count=4,
        hidden_size=16,
        params_dtype=torch.float32,
        hc_lowrank=8,
        rms_norm_eps=1.0e-6,
        hc_per_branch_norm=True,
    )


def test_packed_down_inject_weights_preserve_formula_and_share_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hc, "can_use_hc_triton", None)
    monkeypatch.setattr(hc, "can_use_hc_inject_triton", None)
    torch.manual_seed(11)
    module = hc.GatedResidualSimple(_config())
    hidden = torch.randn(3, module.hyper_hidden_size)
    block_output = torch.randn(3, module.hidden_size)

    mixed_ref, residuals_ref = module.mix(hidden)
    combined_ref = module.combine(block_output, residuals_ref)
    assert module.pack_down_inject_weights()
    assert module._packed_down_inject_weight is not None
    packed = module._packed_down_inject_weight
    down = module.input_mix_weight_down.weight
    inject = module.block_inject_weight.weight
    assert packed.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert packed.untyped_storage().data_ptr() == inject.untyped_storage().data_ptr()
    assert "_packed_down_inject_weight" not in module.state_dict()

    mixed, residuals = module.mix(hidden)
    combined = module.combine(block_output, residuals)
    torch.testing.assert_close(mixed, mixed_ref)
    torch.testing.assert_close(combined, combined_ref)

    # A reload into the original checkpoint names updates the packed storage.
    with torch.no_grad():
        down.fill_(2.0)
        inject.fill_(3.0)
    assert torch.equal(packed[: down.shape[0]], down)
    assert torch.equal(packed[down.shape[0] :], inject)
    assert module.pack_down_inject_weights()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_hc_triton_ops_match_reference_and_replay_graph(
    rows: int, dtype: torch.dtype
) -> None:
    from vllm_fl.models.qwen3_8_flash_next.gpu.ops.hyperconnection import (
        can_use_hc_inject_triton,
        can_use_hc_triton,
    )

    torch.manual_seed(17 + rows)
    device = torch.device("cuda")
    hc_count = 4
    hidden_size = 2560
    hyper_hidden = hc_count * hidden_size
    eps = 1.0e-6
    normed = torch.randn((rows, hyper_hidden), device=device, dtype=dtype)
    weight = torch.randn((hyper_hidden,), device=device, dtype=dtype) * 0.01
    logits = torch.randn_like(normed)
    packed_logits = torch.randn((rows, 320 + hc_count), device=device, dtype=dtype)
    injection_logits = packed_logits[:, 320:]
    block_output = torch.randn((rows, hidden_size), device=device, dtype=dtype)
    residual = torch.randn_like(normed)
    assert can_use_hc_triton(normed, weight, logits, residual)
    assert injection_logits.stride() == (320 + hc_count, 1)
    assert injection_logits.is_contiguous() is (rows == 1)
    assert can_use_hc_inject_triton(injection_logits, block_output, residual)

    grouped = normed.float().view(rows, hc_count, hidden_size)
    norm_ref = (
        (
            grouped
            * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
            * (1.0 + weight.float().view(hc_count, hidden_size))
        )
        .flatten(1)
        .to(dtype)
    )
    gate_ref = (
        (torch.sigmoid(logits.float()).view(rows, hc_count, hidden_size) * grouped)
        .mean(1)
        .to(dtype)
    )
    combine_ref = (
        (
            residual.float().view(rows, hc_count, hidden_size)
            + block_output.float()[:, None, :]
            * (2.0 * torch.sigmoid(injection_logits.float() / hc_count))[:, :, None]
        )
        .flatten(1)
        .to(dtype)
    )

    norm = torch.ops.vllm.qwen4_grouped_gemma_rmsnorm(normed, weight, hc_count, eps)
    gate = torch.ops.vllm.qwen4_hc_gate_reduce(logits, normed, hc_count)
    combined = torch.ops.vllm.qwen4_hc_inject_combine(
        injection_logits, block_output, residual, hc_count
    )
    torch.testing.assert_close(norm, norm_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(gate, gate_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(combined, combine_ref, atol=3e-2, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_norm = torch.ops.vllm.qwen4_grouped_gemma_rmsnorm(
            normed, weight, hc_count, eps
        )
        graph_gate = torch.ops.vllm.qwen4_hc_gate_reduce(logits, normed, hc_count)
        graph_combined = torch.ops.vllm.qwen4_hc_inject_combine(
            injection_logits, block_output, residual, hc_count
        )
    normed.copy_(torch.randn_like(normed))
    logits.copy_(torch.randn_like(logits))
    residual.copy_(torch.randn_like(residual))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_norm).all()
    assert torch.isfinite(graph_gate).all()
    assert torch.isfinite(graph_combined).all()

    first = graph_combined.clone()
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_combined, first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_production_hc_module_fast_path_matches_formula_and_replays_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_fl.models.qwen3_8_flash_next.gpu.ops.hyperconnection import (
        can_use_hc_inject_triton,
        can_use_hc_triton,
    )

    torch.manual_seed(23)
    config = hc.HyperConnectionConfig(
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.bfloat16,
        hc_lowrank=320,
        rms_norm_eps=1.0e-6,
        hc_per_branch_norm=True,
    )
    module = hc.GatedResidualSimple(config).cuda()
    hidden = torch.randn((64, 10240), device="cuda", dtype=torch.bfloat16)
    block_output = torch.randn((64, 2560), device="cuda", dtype=torch.bfloat16)

    monkeypatch.setattr(hc, "can_use_hc_triton", None)
    monkeypatch.setattr(hc, "can_use_hc_inject_triton", None)
    mixed_ref, residuals_ref = module.mix(hidden)
    combined_ref = module.combine(block_output, residuals_ref)
    assert module.pack_down_inject_weights()
    monkeypatch.setattr(hc, "can_use_hc_triton", can_use_hc_triton)
    monkeypatch.setattr(
        hc, "can_use_hc_inject_triton", can_use_hc_inject_triton
    )

    mixed, residuals = module.mix(hidden)
    injection_logits = residuals[2]
    assert injection_logits is not None
    assert injection_logits.stride() == (324, 1)
    assert not injection_logits.is_contiguous()
    assert can_use_hc_inject_triton(injection_logits, block_output, residuals[0])
    combined = module.combine(block_output, residuals)
    torch.testing.assert_close(mixed, mixed_ref, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(combined, combined_ref, atol=5e-2, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_mixed, graph_residuals = module.mix(hidden)
        graph_combined = module.combine(block_output, graph_residuals)
    hidden.copy_(torch.randn_like(hidden))
    block_output.copy_(torch.randn_like(block_output))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_mixed).all()
    assert torch.isfinite(graph_combined).all()
