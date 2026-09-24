# SPDX-License-Identifier: Apache-2.0
"""GPU contract checks for the delayed HC all-optimization candidate.

These tests intentionally exercise the combined path only.  They compare the
fused combine+norm boundary with a literal rounded reference, keep the packed
injection view's widened row stride, and replay the same graph with updated
static inputs.  Decoder/PLE/MTP wiring is checked from the source contract so
this file does not silently construct a second model configuration.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.common.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for HC allopt checks"
)

_METRICS: dict[str, dict[str, float | bool]] = {}


def _record_error(name: str, output: torch.Tensor, reference: torch.Tensor) -> None:
    diff = (output.float() - reference.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    rmse = float(diff.square().mean().sqrt().item()) if diff.numel() else 0.0
    ref_max = float(reference.float().abs().max().item()) if reference.numel() else 0.0
    _METRICS[name] = {
        "max_abs": max_abs,
        "rmse": rmse,
        "reference_max_abs": ref_max,
        "all_finite": bool(torch.isfinite(output).all().item()),
    }
    print(f"[hc-metric] {name}: max_abs={max_abs:.7g} rmse={rmse:.7g}")
    metrics_path = os.environ.get("HC_ALLOPT_METRICS_PATH")
    if metrics_path:
        Path(metrics_path).write_text(
            json.dumps({"metrics": _METRICS}, indent=2, sort_keys=True) + "\n"
        )


def _reference_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = residual.shape
    hidden_size = dim // hc_count
    weight = norm_weight.float()
    combined = (
        residual.float().view(rows, hc_count, hidden_size)
        + block_output.float()[:, None, :]
        * (2.0 * torch.sigmoid(injection_logits.float() / hc_count))[:, :, None]
    ).to(residual.dtype)
    grouped = combined.float().view(rows, hc_count, hidden_size)
    inv_rms = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    if weight.numel() == hidden_size:
        affine = weight.view(1, 1, hidden_size)
    else:
        affine = weight.view(1, hc_count, hidden_size)
    normalized = (grouped * inv_rms * (1.0 + affine)).flatten(1).to(residual.dtype)
    return combined.flatten(1), normalized


def _inputs(
    rows: int, dtype: torch.dtype, *, shared_weight: bool
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(701 + rows + int(shared_weight))
    hc_count = 4
    hidden_size = 2560
    residual = torch.randn(rows, hc_count * hidden_size, device="cuda", dtype=dtype)
    block = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    # The injection view is a split from the packed 320+HC projection.  It is
    # contiguous in its last dimension but keeps a widened row stride.
    packed = torch.randn(rows, 320 + hc_count, device="cuda", dtype=dtype)
    injection = packed[:, 320:]
    assert injection.stride() == (324, 1)
    weight = torch.randn(
        hidden_size if shared_weight else hc_count * hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.01
    return residual, block, injection, weight


@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("shared_weight", [True, False])
def test_combine_norm_rounding_and_packed_stride(
    rows: int, shared_weight: bool
) -> None:
    from vllm_fl.models.qwen3_8_flash_next.gpu.ops.hyperconnection import (
        can_use_hc_combine_norm_triton,
    )

    dtype = torch.bfloat16
    residual, block, injection, weight = _inputs(
        rows, dtype, shared_weight=shared_weight
    )
    assert can_use_hc_combine_norm_triton(injection, block, residual, weight)
    eps = 1.0e-6
    out, normed = torch.ops.vllm.qwen4_hc_combine_norm(
        residual, block, injection, weight, eps, 4
    )
    ref_out, ref_normed = _reference_combine_norm(
        residual, block, injection, weight, eps, 4
    )
    if weight.numel() == residual.shape[-1]:
        baseline_out = torch.ops.vllm.qwen4_hc_inject_combine(
            injection, block, residual, 4
        )
        baseline_normed = torch.ops.vllm.qwen4_grouped_gemma_rmsnorm(
            baseline_out, weight, 4, eps
        )
        _record_error(
            f"baseline_combine_rows{rows}_shared{shared_weight}",
            baseline_out,
            ref_out,
        )
        _record_error(
            f"baseline_norm_rows{rows}_shared{shared_weight}",
            baseline_normed,
            ref_normed,
        )
    _record_error(f"fused_combine_rows{rows}_shared{shared_weight}", out, ref_out)
    _record_error(f"fused_norm_rows{rows}_shared{shared_weight}", normed, ref_normed)
    torch.testing.assert_close(out, ref_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(normed, ref_normed, atol=4.0e-2, rtol=3.0e-2)
    assert out.dtype is dtype
    assert normed.dtype is dtype
    assert out.is_contiguous()
    assert normed.is_contiguous()


def test_combine_norm_cuda_graph_replay_is_deterministic() -> None:
    residual, block, injection, weight = _inputs(5, torch.bfloat16, shared_weight=False)
    eps = 1.0e-6

    # Eager warmup compiles/lazily allocates before capture.
    eager = torch.ops.vllm.qwen4_hc_combine_norm(
        residual, block, injection, weight, eps, 4
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_normed = torch.ops.vllm.qwen4_hc_combine_norm(
            residual, block, injection, weight, eps, 4
        )
    first_inputs = [x.clone() for x in (residual, block, injection, weight)]
    first_ref = _reference_combine_norm(*first_inputs, eps, 4)
    _record_error("graph_warmup_combined", eager[0], first_ref[0])
    _record_error("graph_warmup_normed", eager[1], first_ref[1])
    torch.testing.assert_close(
        eager[0], first_ref[0], atol=2.0e-2, rtol=2.0e-2
    )

    for iteration in range(10):
        residual.copy_(torch.randn_like(residual))
        block.copy_(torch.randn_like(block))
        injection.copy_(torch.randn_like(injection))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference_combine_norm(residual, block, injection, weight, eps, 4)
        if iteration == 0:
            _record_error("graph_replay_combined", graph_out, ref[0])
            _record_error("graph_replay_normed", graph_normed, ref[1])
        torch.testing.assert_close(
            graph_out, ref[0], atol=2.0e-2, rtol=2.0e-2, msg=f"output iter {iteration}"
        )
        torch.testing.assert_close(
            graph_normed,
            ref[1],
            atol=4.0e-2,
            rtol=3.0e-2,
            msg=f"norm iter {iteration}",
        )
        if iteration == 0:
            first_graph_out = graph_out.clone()
        else:
            # The input changes, so deterministic means repeat the same input
            # once more and compare exact output, not compare different inputs.
            pass

    residual.copy_(torch.full_like(residual, 0.125))
    block.copy_(torch.full_like(block, -0.25))
    injection.copy_(torch.full_like(injection, 0.5))
    graph.replay()
    torch.cuda.synchronize()
    repeat = graph_out.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, repeat)
    del first_graph_out


def test_delayed_module_matches_eager_and_final_mixer() -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=256,
        params_dtype=torch.bfloat16,
        hc_lowrank=32,
        rms_norm_eps=1.0e-6,
        hc_per_branch_norm=True,
    )
    module = GatedResidualSimple(config).cuda()
    final = GatedResidualSimple(config, use_combine=False, role="final").cuda()
    module.pack_down_inject_weights()
    torch.manual_seed(733)
    hidden = torch.randn(7, module.hyper_hidden_size, device="cuda", dtype=torch.bfloat16)
    block = torch.randn(7, module.hidden_size, device="cuda", dtype=torch.bfloat16)

    _, eager_residuals = module.mix(hidden)
    eager_combined = module.combine(block, eager_residuals)
    eager_normed = module._normalize(eager_combined)
    eager_next, eager_injection = module._mix_from_normed(eager_normed)

    delayed_state, delayed_input, injection = module.mix_delayed(hidden)
    delayed_combined, delayed_next, delayed_injection = module.combine_and_mix(
        delayed_state, block, injection
    )
    torch.testing.assert_close(delayed_combined, eager_combined, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(delayed_next, eager_next, atol=7e-2, rtol=4e-2)
    torch.testing.assert_close(delayed_injection, eager_injection, atol=5e-2, rtol=3e-2)
    _record_error("module_delayed_combined_h256", delayed_combined, eager_combined)
    _record_error("module_delayed_input_h256", delayed_next, eager_next)
    _record_error("module_delayed_injection_h256", delayed_injection, eager_injection)
    assert delayed_input.shape == (7, module.hidden_size)

    # The final mixer has no new injection projection but must consume the
    # pending tuple and still return a sampled [M,H] block input.
    final_state, final_input, final_injection = final.combine_and_mix(
        hidden, block, injection
    )
    final_ref = module._normalize(eager_combined)
    torch.testing.assert_close(final_state, eager_combined, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(final_input, final._mix_from_normed(final_ref)[0], atol=7e-2, rtol=4e-2)
    assert final_injection is None


def test_production_c64_h2560_delayed_combine_and_mix_cuda_graph() -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.bfloat16,
        hc_lowrank=320,
        rms_norm_eps=1.0e-6,
        hc_per_branch_norm=True,
    )
    module = GatedResidualSimple(config).cuda()
    assert module.pack_down_inject_weights()
    rows = 64
    torch.manual_seed(787)
    hidden = torch.randn(rows, module.hyper_hidden_size, device="cuda", dtype=torch.bfloat16)
    block = torch.randn(rows, module.hidden_size, device="cuda", dtype=torch.bfloat16)
    state, _, injection = module.mix_delayed(hidden)
    assert state.data_ptr() == hidden.data_ptr()
    assert injection is not None
    assert injection.shape == (rows, config.hc_count)
    assert injection.stride() == (config.hc_lowrank + config.hc_count, 1)
    # Keep the production split's widened stride, but move the graph input to
    # independent storage.  The source split carries an autograd multi-view
    # history; mutating it between graph replays would invalidate that view.
    split_injection = injection
    injection = torch.empty_strided(
        split_injection.shape,
        split_injection.stride(),
        dtype=split_injection.dtype,
        device=split_injection.device,
    )
    injection.copy_(split_injection.detach())
    assert injection.stride() == (config.hc_lowrank + config.hc_count, 1)

    eager = module.combine_and_mix(state, block, injection)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_state, graph_input, graph_injection = module.combine_and_mix(
            state, block, injection
        )
    torch.cuda.synchronize()

    for iteration in range(2):
        hidden.copy_(torch.randn_like(hidden))
        block.copy_(torch.randn_like(block))
        # ``mix_delayed`` intentionally returns the packed projection's
        # split view.  Autograd forbids a plain in-place update on a
        # multi-view result, while the CUDA-graph input update itself is
        # valid under no_grad (and preserves the widened (324, 1) stride).
        with torch.no_grad():
            injection.copy_(torch.randn_like(injection))
        graph.replay()
        torch.cuda.synchronize()
        expected = module.combine_and_mix(state, block, injection)
        if iteration == 0:
            _record_error("production_c64_state", graph_state, expected[0])
            _record_error("production_c64_input", graph_input, expected[1])
            _record_error("production_c64_injection", graph_injection, expected[2])
        torch.testing.assert_close(graph_state, expected[0], atol=8e-2, rtol=5e-2)
        torch.testing.assert_close(graph_input, expected[1], atol=8e-2, rtol=5e-2)
        torch.testing.assert_close(graph_injection, expected[2], atol=8e-2, rtol=5e-2)
    assert eager[0].shape == (rows, config.hc_count * config.hidden_size)


def test_delayed_combine_and_mix_cuda_graph_replay() -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=256,
        params_dtype=torch.bfloat16,
        hc_lowrank=32,
        rms_norm_eps=1.0e-6,
        hc_per_branch_norm=True,
    )
    module = GatedResidualSimple(config).cuda()
    module.pack_down_inject_weights()
    rows = 7
    hidden = torch.randn(
        rows, module.hyper_hidden_size, device="cuda", dtype=torch.bfloat16
    )
    block = torch.randn(rows, module.hidden_size, device="cuda", dtype=torch.bfloat16)
    packed_injection = torch.randn(
        rows,
        config.hc_lowrank + config.hc_count,
        device="cuda",
        dtype=torch.bfloat16,
    )
    injection = packed_injection[:, config.hc_lowrank :]
    assert injection.stride() == (config.hc_lowrank + config.hc_count, 1)

    # Warm up the complete delayed boundary before capture.  The captured
    # graph includes combine_norm, packed projection split, SiLU and gate_reduce.
    eager = module.combine_and_mix(hidden, block, injection)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_state, graph_input, graph_injection = module.combine_and_mix(
            hidden, block, injection
        )

    for iteration in range(5):
        hidden.copy_(torch.randn_like(hidden))
        block.copy_(torch.randn_like(block))
        packed_injection.copy_(torch.randn_like(packed_injection))
        graph.replay()
        torch.cuda.synchronize()
        expected = module.combine_and_mix(hidden, block, injection)
        torch.testing.assert_close(
            graph_state,
            expected[0],
            atol=7.0e-2,
            rtol=4.0e-2,
            msg=f"state iter {iteration}",
        )
        torch.testing.assert_close(
            graph_input,
            expected[1],
            atol=8.0e-2,
            rtol=5.0e-2,
            msg=f"input iter {iteration}",
        )
        torch.testing.assert_close(
            graph_injection,
            expected[2],
            atol=7.0e-2,
            rtol=4.0e-2,
            msg=f"injection iter {iteration}",
        )
    assert eager[0].shape == graph_state.shape


def test_combine_norm_torch_compile_fake_contract() -> None:
    rows = 2
    hc_count = 4
    hidden_size = 256
    dtype = torch.bfloat16
    residual = torch.randn(rows, hc_count * hidden_size, device="cuda", dtype=dtype)
    block = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    packed = torch.randn(rows, 32 + hc_count, device="cuda", dtype=dtype)
    injection = packed[:, 32:]
    weight = torch.randn(hc_count * hidden_size, device="cuda", dtype=dtype) * 0.01
    eps = 1.0e-6

    def combine_norm(
        x: torch.Tensor,
        b: torch.Tensor,
        i: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.qwen4_hc_combine_norm(x, b, i, w, eps, hc_count)

    compiled = torch.compile(combine_norm, dynamic=False)
    eager = combine_norm(residual, block, injection, weight)
    compiled_out = compiled(residual, block, injection, weight)
    torch.testing.assert_close(compiled_out[0], eager[0], atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(compiled_out[1], eager[1], atol=4.0e-2, rtol=3.0e-2)


def test_decoder_ple_mtp_pending_contract_is_present() -> None:
    root = Path(__file__).parents[2] / "vllm_fl/models/qwen3_8_flash_next/gpu"
    model_source = (root / "model.py").read_text()
    mtp_source = (root / "mtp.py").read_text()
    for source in (model_source, mtp_source):
        ast.parse(source)
        assert "prev_block_output" in source
        assert "prev_injection" in source
        assert "combine_pending" in source
        assert "combine_and_mix" in source
    assert "block_output = injection = None" in model_source
    assert "last_layer.mlp_hyper_connection.combine_pending" in model_source
    assert "self.hyper_connection_mixer.combine_and_mix" in mtp_source
