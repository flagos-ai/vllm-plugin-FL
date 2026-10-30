# SPDX-License-Identifier: Apache-2.0
"""Exercise the ModelSlim loader through the actual upstream modular kernel."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


@pytest.fixture
def workspace():
    import torch_npu  # noqa: F401

    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from vllm_fl import _patch_ascend_torch_accelerator

    # Match the existing general-plugin startup before allocating workspaces.
    _patch_ascend_torch_accelerator()
    assert torch.accelerator.empty_cache is torch.npu.empty_cache
    init_workspace_manager(torch.device("npu:0"))
    try:
        yield
    finally:
        reset_workspace_manager()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_loader_modular_eager_and_graph(dtype, workspace):
    import torch_npu  # noqa: F401

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    from vllm_fl.dispatch import get_default_manager, with_strict_mode
    from vllm_fl.dispatch.backends.reference.impl.w8a8_moe import (
        modelslim_w8a8_experts,
    )
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimMoEMethod

    config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=64,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="npu",
        in_dtype=dtype,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = AscendModelSlimMoEMethod(config)
    layer = torch.nn.Module()
    layer._expert_routing_tables = lambda: None
    layer.activation = config.activation
    layer.global_num_experts = 2
    layer.expert_map = None
    layer.apply_router_weight_on_input = False

    def loader(*args):
        pass

    method.create_weights(layer, 2, 128, 64, dtype, weight_loader=loader)
    assert layer.w13_weight.shape == (2, 128, 128)
    assert layer.w2_weight.shape == (2, 128, 64)
    assert layer.w13_weight_scale.shape == (2, 128, 1)
    assert layer.w13_weight_scale.quant_method == "channel"
    assert all(param.weight_loader is loader for param in layer.parameters())
    layer = layer.to("npu")
    torch.manual_seed(19)
    with torch.no_grad():
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(layer, name)
            weight.copy_(
                torch.randint(-24, 24, weight.shape, dtype=torch.int8, device="npu")
            )
        layer.w13_weight_scale.fill_(0.015625)
        layer.w2_weight_scale.fill_(0.015625)
    w1, w2 = layer.w13_weight.detach().clone(), layer.w2_weight.detach().clone()
    s1 = layer.w13_weight_scale.detach().clone().float()
    s2 = layer.w2_weight_scale.detach().clone().float()
    x = torch.randn(3, 128, dtype=dtype, device="npu")
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device="npu", dtype=torch.int32)
    probabilities = torch.full((3, 2), 0.5, device="npu")

    def apply():
        return method.apply(layer, x, probabilities, ids, None, None)

    with with_strict_mode():
        manager = get_default_manager()
        for op in ("modelslim_w8a8_prepare_experts", "modelslim_w8a8_experts"):
            assert manager.get_selected_impl_id(op) == "vendor.ascend"
        method.process_weights_after_loading(layer)
        assert method.supports_internal_mk
        assert method.moe_kernel.fused_experts.expects_unquantized_inputs
        expected = modelslim_w8a8_experts(x, w1, w2, s1, s2, probabilities, ids)
        torch.testing.assert_close(apply(), expected, rtol=0.03, atol=0.06)
        for _ in range(3):
            apply()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = apply()
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            expected = apply()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected)
