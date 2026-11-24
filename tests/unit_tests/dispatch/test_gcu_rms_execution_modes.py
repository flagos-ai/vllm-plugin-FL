"""GCU inference must use one RMS provider across eager/compiled graphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.gcu.impl.normalization import rms_norm_gcu


def test_inference_eager_matches_provider_and_replay(monkeypatch):
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("GCU runtime is unavailable")
    # Non-power-of-two width exercises masked reductions; batch size is not a
    # multiple of the kernel's row tile. Preserve inputs for repeated replay.
    torch.manual_seed(119)
    x = torch.randn(5, 127, device="gcu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    obj = SimpleNamespace(
        weight=torch.randn(127, device="gcu", dtype=x.dtype),
        variance_epsilon=1e-6,
    )
    from vllm_fl.dispatch.backends.vendor.gcu.impl import normalization
    from vllm_fl.dispatch.backends.vendor.gcu.impl.normalization import _rms_op

    calls = []

    def tracked_provider(*args):
        calls.append(True)
        return _rms_op(*args)

    monkeypatch.setattr(normalization, "_rms_op", tracked_provider)
    with torch.inference_mode():
        eager = rms_norm_gcu(obj, x, residual)
        assert calls, "Eager inference must use the graph RMS provider"
        compiled_provider = _rms_op(x, obj.weight, residual, 127, 1e-6, True)
        for actual, expected in zip(eager, compiled_provider):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.gcu.synchronize()
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            captured = rms_norm_gcu(obj, x, residual)
        for _ in range(2):
            graph.replay()
            torch.gcu.synchronize()
            for actual, expected in zip(captured, eager):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
