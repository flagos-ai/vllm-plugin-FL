# SPDX-License-Identifier: Apache-2.0
"""Model-local small BF16 projection compatibility under full FlagGems dispatch."""

import math

import flag_gems
import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax


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
    eager_error = (eager.float() - expected.float()).abs()
    graph_error = (captured.float() - expected_changed.float()).abs()
    relative_l2 = float(
        (captured.float() - expected_changed.float()).norm()
        / expected_changed.float().norm().clamp_min(1e-12)
    )
    print(
        {
            "M": m,
            "K": k,
            "N": n,
            "eager_max_abs": float(eager_error.max()),
            "graph_max_abs": float(graph_error.max()),
            "graph_relative_l2": relative_l2,
            "projection_dtype_before_hc_division": str(captured.dtype),
            "flag_gems_context": True,
        }
    )
