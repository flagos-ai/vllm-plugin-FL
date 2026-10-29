# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise real dispatch and the vLLM router on the available device.

Run this file unchanged on the target vendor image. CPU runs verify the
reference integration; the graph test requires an actual accelerator.
"""

import pytest
import torch


@pytest.fixture
def routing_dispatch(device, has_accelerator):
    from vllm_fl.dispatch.manager import reset_default_manager
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    assert not has_accelerator or device.type != "cpu", (
        "The configured accelerator mapped to CPU in the device fixture; "
        "fix the platform/device configuration before claiming a device run."
    )
    reset_default_manager()
    with policy_context(SelectionPolicy(prefer="vendor", strict=True)):
        yield
    reset_default_manager()


@pytest.mark.parametrize(
    "tokens,experts,topk,dtype,renormalize",
    [
        (0, 8, 2, torch.float32, False),
        (1, 8, 1, torch.float16, False),
        (7, 8, 8, torch.float32, True),
        (7, 64, 2, torch.bfloat16, True),
    ],
)
def test_real_dispatch_matches_stable_score_oracle(
    routing_dispatch, device, tokens, experts, topk, dtype, renormalize
):
    from vllm import _custom_ops  # noqa: F401 - loads the real compiled schemas
    from vllm.platforms import current_platform

    from vllm_fl.dispatch import get_default_manager

    # Repeated logits deliberately exercise deterministic ties. Bias changes
    # ranking while the independent oracle gathers weights from unbiased scores.
    logits = (torch.arange(tokens * experts).reshape(tokens, experts) % 5 - 2).to(
        dtype=dtype, device=device
    )
    bias = ((torch.arange(experts) % 3) * 0.375).to(device)
    weights = torch.empty(tokens, topk, dtype=torch.float32, device=device)
    ids = torch.empty(tokens, topk, dtype=torch.int32, device=device)
    rows = torch.empty_like(ids)
    manager = get_default_manager()
    fn = manager.resolve("topk_sigmoid")
    returned = fn(weights, ids, rows, logits, renormalize, bias)

    scores = logits.cpu().float().sigmoid()
    expected_ids = (scores + bias.cpu()).argsort(dim=-1, descending=True, stable=True)[
        :, :topk
    ]
    expected_weights = scores.gather(-1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    expected_rows = torch.arange(topk)[None, :] * tokens + torch.arange(tokens)[:, None]
    assert returned[0] is weights and returned[1] is ids
    torch.testing.assert_close(ids.cpu().long(), expected_ids)
    torch.testing.assert_close(weights.cpu(), expected_weights, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows.cpu().long(), expected_rows)

    # This assertion distinguishes real native dispatch from a numerically
    # correct reference fallback on an NVIDIA installation with the compiled op.
    native_installed = getattr(torch.ops._moe_C, "topk_sigmoid", None) is not None
    native_expected = (
        torch.cuda.is_available()
        and getattr(current_platform, "vendor_name", None) == "nvidia"
        and native_installed
    )
    expected_impl = "vendor.cuda" if native_expected else "reference.torch"
    registered = manager.registry.get_implementation("topk_sigmoid", expected_impl)
    assert registered is not None
    assert fn is registered.fn


def test_real_vllm_router_preserves_bias_indices_and_scaling(routing_dispatch, device):
    from vllm_fl.ops.fused_moe.router import FusedTopKBiasRouterFL

    router = FusedTopKBiasRouterFL(
        top_k=2,
        global_num_experts=4,
        e_score_correction_bias=torch.tensor([0.0, -1.0, 2.0, 0.0], device=device),
        renormalize=True,
        routed_scaling_factor=1.5,
        scoring_func="sigmoid",
    )
    weights, ids = router._compute_routing(
        torch.empty(1, 8, device=device),
        torch.tensor([[0.0, 1.0, -1.0, 2.0]], device=device),
        torch.int64,
    )
    assert ids.dtype == torch.int64
    assert ids.cpu().tolist() == [[2, 3]]
    selected = torch.tensor([-1.0, 2.0]).sigmoid()
    expected = (selected / selected.sum() * 1.5).unsqueeze(0)
    torch.testing.assert_close(weights.cpu(), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.gpu
def test_sigmoid_dispatch_graph_capture_and_replay(routing_dispatch, device):
    from vllm_fl.dispatch import CachedOp, prewarm_cached_ops

    runtime = getattr(torch, device.type, None)
    graph_type = "NPUGraph" if device.type == "npu" else "CUDAGraph"
    if (
        runtime is None
        or not hasattr(runtime, graph_type)
        or not hasattr(runtime, "graph")
    ):
        pytest.skip(f"No supported graph API for {device.type}")

    logits = torch.zeros(2, 8, device=device)
    weights = torch.empty(2, 2, dtype=torch.float32, device=device)
    ids = torch.empty(2, 2, dtype=torch.int32, device=device)
    rows = torch.empty_like(ids)
    op = CachedOp("topk_sigmoid")
    prewarm_cached_ops()
    stream = runtime.Stream()
    stream.wait_stream(runtime.current_stream())
    with runtime.stream(stream):
        for _ in range(3):
            op(weights, ids, rows, logits, True, None)
    runtime.current_stream().wait_stream(stream)
    runtime.synchronize()
    graph = getattr(runtime, graph_type)()
    with runtime.graph(graph):
        op(weights, ids, rows, logits, True, None)

    # Replaying with changed input catches a capture that returned stale outputs.
    logits[:, 3] = 2.0
    graph.replay()
    runtime.synchronize()
    assert ids.cpu().tolist() == [[3, 0], [3, 0]]
    selected = torch.tensor([2.0, 0.0]).sigmoid()
    expected = (selected / selected.sum()).expand(2, -1)
    torch.testing.assert_close(weights.cpu(), expected, atol=1e-6, rtol=1e-5)
