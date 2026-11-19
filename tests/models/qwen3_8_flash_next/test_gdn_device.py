# ruff: noqa: E402
# Optional device imports follow the explicit opt-in gate.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

from __future__ import annotations

import math
import os

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip("requires an assigned MetaX device", allow_module_level=True)

import flag_gems
import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax

if not torch.cuda.is_available():
    raise RuntimeError("QWEN38_RUN_DEVICE_TESTS=1 requires an available accelerator")

from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as adapter_ops
from vllm_fl.dispatch.backends.vendor.metax.impl import (
    qwen3_8_flash_next as adapter_metax,
)

# Adapter contracts.

# SPDX-License-Identifier: Apache-2.0


pytestmark = pytest.mark.gpu


def adapter__packed_oracle(mixed, a, b, A_log, dt_bias, state, indices):
    mixed, a, b = mixed.cpu().float(), a.cpu().float(), b.cpu().float()
    A_log, dt_bias = A_log.cpu().float(), dt_bias.cpu().float()
    state, indices = state.cpu().clone(), indices.cpu()
    rows = mixed.shape[0]
    hv, vd, kd = state.shape[-3:]
    h = (mixed.shape[1] - hv * vd) // (2 * kd)
    q, k, v = mixed.split((h * kd, h * kd, hv * vd), dim=-1)
    q, k, v = q.view(rows, h, kd), k.view(rows, h, kd), v.view(rows, hv, vd)
    out = torch.zeros(rows, 1, hv, vd)
    for row in range(rows):
        slot = int(indices[row])
        if slot <= 0:
            continue
        qr = q[row] / torch.sqrt(q[row].square().sum(-1, keepdim=True) + 1e-6)
        kr = k[row] / torch.sqrt(k[row].square().sum(-1, keepdim=True) + 1e-6)
        qr = qr.repeat_interleave(hv // h, 0) * kd**-0.5
        kr = kr.repeat_interleave(hv // h, 0)
        x = a[row] + dt_bias
        softplus = torch.where(x <= 20, torch.log(1 + torch.exp(x)), x)
        g = -torch.exp(A_log) * softplus
        beta = torch.sigmoid(b[row])
        updated = state[slot] * torch.exp(g)[:, None, None]
        delta = (v[row] - (updated * kr[:, None, :]).sum(-1)) * beta[:, None]
        updated += delta[:, :, None] * kr[:, None, :]
        out[row, 0] = (updated * qr[:, None, :]).sum(-1)
        state[slot] = updated
    return out.to(torch.bfloat16), state


def adapter__assert_packed(mixed, a, b, alog, bias, state, indices, out, before):
    expected_out, expected_state = adapter__packed_oracle(
        mixed, a, b, alog, bias, before, indices
    )
    actual_state = state.cpu()
    state_error = float((actual_state - expected_state).abs().max())
    torch.testing.assert_close(out.cpu(), expected_out, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(actual_state[0], before.cpu()[0], atol=0, rtol=0)
    return state_error


def test_packed_gdn_fp32_beta_eager_and_changed_input_graph():
    import flaggems_vllm

    torch.manual_seed(721)
    rows, h, hv, d = 3, 2, 6, 128
    mixed = torch.randn(rows, (2 * h + hv) * d, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(rows, hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1 - 1.0
    bias = torch.randn(hv, device="cuda") * 0.1
    initial = torch.randn(5, hv, d, d, device="cuda") * 0.03
    state = initial.clone()
    indices = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32)
    out = torch.empty(rows, 1, hv, d, device="cuda", dtype=torch.bfloat16)

    def run():
        return adapter_ops.gdn_packed_decode(
            mixed, a, b, alog, bias, d**-0.5, state, out, indices, True
        )

    run()
    eager_error = adapter__assert_packed(
        mixed, a, b, alog, bias, state, indices, out, initial
    )

    # The independent oracle must distinguish the uncorrected BF16-beta ABI.
    old_state, old_out = initial.clone(), torch.empty_like(out)
    flaggems_vllm.fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, alog, bias, d**-0.5, old_state, old_out, indices, True
    )
    _, reference_state = adapter__packed_oracle(
        mixed, a, b, alog, bias, initial, indices
    )
    old_error = float((old_state.cpu() - reference_state).abs().max())
    assert old_error > max(1e-4, eager_error * 3), (old_error, eager_error)

    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    mixed.normal_()
    a.normal_()
    b.normal_()
    indices.copy_(torch.tensor([2, 1, -1], device="cuda", dtype=torch.int32))
    state.copy_(initial)
    graph.replay()
    adapter__assert_packed(mixed, a, b, alog, bias, state, indices, out, initial)


def test_gdn_prefill_preserves_fp32_vmajor_state():
    torch.manual_seed(913)
    t, h, hv, d = 8, 2, 6, 128
    q = torch.randn(1, t, h, d)
    k = torch.randn_like(q)
    q = (q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    v = (torch.randn(1, t, hv, d) * 0.3).bfloat16()
    g = -torch.rand(1, t, hv) * 0.5
    beta = torch.rand(1, t, hv)
    initial = torch.randn(1, hv, d, d) * 0.01
    reference_state = initial.clone()
    reference_output = torch.empty_like(v)
    for token in range(t):
        qr = q[0, token].float().repeat_interleave(hv // h, 0) * d**-0.5
        kr = k[0, token].float().repeat_interleave(hv // h, 0)
        state = reference_state[0] * g[0, token].exp()[:, None, None]
        delta = v[0, token].float() - (state * kr[:, None, :]).sum(-1)
        delta *= beta[0, token, :, None]
        state += delta[:, :, None] * kr[:, None, :]
        reference_output[0, token] = (state * qr[:, None, :]).sum(-1)
        reference_state[0] = state
    output, final = adapter_metax.gdn_prefill(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        g.cuda(),
        beta.cuda(),
        initial.cuda(),
        True,
        cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device="cuda"),
        chunk_indices=torch.tensor([[0, 0]], dtype=torch.int32, device="cuda"),
        chunk_offsets=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=False,
    )
    assert final.dtype == torch.float32
    torch.testing.assert_close(output.cpu(), reference_output, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(final.cpu(), reference_state, atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("m", [2048, 1, 2, 8])
@pytest.mark.parametrize("n,k", [(4, 10240), (1, 2560)])
def test_projection_eager_and_changed_input_graph_under_flaggems(m, n, k):
    torch.manual_seed(814 + m + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    changed_x = torch.randn_like(x)
    changed_weight = torch.randn_like(weight) / math.sqrt(k)
    assert getattr(flag_gems, "current_work_registrar", None) is None
    # Both BF16 native Torch references are outside FlagGems registration.
    expected = F.linear(x, weight)
    expected_changed = F.linear(changed_x, changed_weight)
    expected_divided = expected / 4 if n == 4 else expected
    expected_changed_divided = expected_changed / 4 if n == 4 else expected_changed

    with flag_gems.use_gems():
        assert getattr(flag_gems, "current_work_registrar", None) is not None
        eager = metax.qwen38_small_projection(x, weight)
        eager_divided = eager / 4 if n == 4 else eager
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = metax.qwen38_small_projection(x, weight)
            captured_divided = captured / 4 if n == 4 else captured
        x.copy_(changed_x)
        weight.copy_(changed_weight)
        captured.fill_(float("nan"))
        graph.replay()

    assert eager.dtype == torch.bfloat16
    assert captured.dtype == torch.bfloat16
    assert eager_divided.dtype == torch.bfloat16
    assert captured_divided.dtype == torch.bfloat16
    torch.testing.assert_close(eager, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(captured, expected_changed, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(eager_divided, expected_divided, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(
        captured_divided, expected_changed_divided, atol=3e-3, rtol=1e-2
    )


def test_strided_qkv_gather_eager_and_changed_input_graph_under_flaggems():
    import flag_gems

    from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax

    torch.manual_seed(942)
    projected = torch.randn(42, 2048, device="cuda", dtype=torch.bfloat16)
    changed = torch.randn_like(projected)
    qkv = projected[:, :1280]
    assert qkv.shape == (42, 1280) and qkv.stride() == (2048, 1)
    spec_indices = torch.arange(2, device="cuda", dtype=torch.int64)
    prefill_indices = torch.arange(2, 42, device="cuda", dtype=torch.int64)
    assert getattr(flag_gems, "current_work_registrar", None) is None
    # Native Torch references stay outside the FlagGems registration context.
    expected = tuple(
        qkv.index_select(0, index) for index in (spec_indices, prefill_indices)
    )
    expected_changed = tuple(
        changed[:, :1280].index_select(0, index)
        for index in (spec_indices, prefill_indices)
    )

    def gather():
        packed = metax.qwen38_gdn_mixed_qkv(qkv)
        assert packed.is_contiguous()
        return tuple(
            packed.index_select(0, index) for index in (spec_indices, prefill_indices)
        )

    with flag_gems.use_gems():
        assert getattr(flag_gems, "current_work_registrar", None) is not None
        eager = gather()
        for _ in range(2):
            gather()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gather()
        projected.copy_(changed)
        for value in captured:
            value.fill_(float("nan"))
        graph.replay()

    for actual, reference in zip(eager, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for actual, reference in zip(captured, expected_changed):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
