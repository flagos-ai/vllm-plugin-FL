# SPDX-License-Identifier: Apache-2.0
"""Run installed-package checks with torchrun --nproc-per-node=2 -m pytest -s.

This is a component fixture using real modular kernels and HCCL groups. The
lightweight weight container is deliberately not a full model or RoutedExperts.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.npu.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
        reason="Requires torchrun with two Ascend processes",
    ),
]


def _record(event, **fields):
    record = {"event": event, "rank": int(os.environ["RANK"]), **fields}
    print(json.dumps(record), flush=True)


@pytest.fixture(scope="module")
def groups():
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dp_group,
        get_ep_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from vllm_fl import _patch_ascend_torch_accelerator
    from vllm_fl.distributed.device_communicators.npu_communicator import (
        NPUCommunicator,
    )

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    _patch_ascend_torch_accelerator()
    config = VllmConfig(
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=2,
            data_parallel_rank=rank,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            all2all_backend="allgather_reducescatter",
        ),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2, rank=rank, local_rank=local_rank, backend="hccl"
        )
        initialize_model_parallel(1, backend="hccl")
        init_workspace_manager(torch.device(f"npu:{local_rank}"))
        try:
            ep, dp, tp = get_ep_group(), get_dp_group(), get_tp_group()
            assert ep.world_size == dp.world_size == 2
            assert tp.world_size == 1
            comm = ep.device_communicator
            assert isinstance(comm, NPUCommunicator)
            assert isinstance(comm.all2all_manager, AgRsAll2AllManager)
            assert isinstance(dp.device_communicator, NPUCommunicator)
            _record(
                "groups",
                groups={
                    name: {
                        "ranks": group.ranks,
                        "world_size": group.world_size,
                        "rank_in_group": group.rank_in_group,
                    }
                    for name, group in (("ep", ep), ("dp", dp), ("tp", tp))
                },
                communicator=type(comm).__name__,
                manager=type(comm.all2all_manager).__name__,
            )
            yield config, ep, dp, tp
        finally:
            try:
                reset_workspace_manager()
            finally:
                try:
                    destroy_model_parallel()
                finally:
                    destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("counts", [(1, 3), (0, 2)])
def test_dynamic_experts_real_agrs(groups, dtype, counts):
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    from vllm_fl.dispatch import get_default_manager, with_strict_mode
    from vllm_fl.dispatch.backends.reference.impl.w8a8_moe import modelslim_w8a8_experts
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimMoEMethod

    vllm_config, ep, dp, tp = groups
    parallel = FusedMoEParallelConfig(
        tp_size=tp.world_size,
        pcp_size=1,
        dp_size=dp.world_size,
        ep_size=ep.world_size,
        tp_rank=tp.rank_in_group,
        pcp_rank=0,
        dp_rank=dp.rank_in_group,
        ep_rank=ep.rank_in_group,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=64,
        num_local_experts=2,
        num_logical_experts=4,
        activation=MoEActivation.SILU,
        device=torch.device("npu", torch.npu.current_device()),
        in_dtype=dtype,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
    )
    method = AscendModelSlimMoEMethod(config)
    layer = torch.nn.Module()
    layer._expert_routing_tables = lambda: None
    layer.activation = config.activation
    layer.global_num_experts = 4
    layer.apply_router_weight_on_input = False
    expert_map = torch.full((4,), -1, dtype=torch.int32, device="npu")
    first = ep.rank_in_group * 2
    expert_map[first : first + 2] = torch.arange(2, device="npu", dtype=torch.int32)
    layer.expert_map = expert_map

    def loader(*args):
        pass

    method.create_weights(layer, 2, 128, 64, dtype, weight_loader=loader)
    assert all(p.weight_loader is loader for p in layer.parameters())
    layer = layer.to("npu")
    # Identical full weights on both ranks, independent of NPU RNG state.
    w1 = (torch.arange(4 * 128 * 128, device="npu").reshape(4, 128, 128) % 47 - 23).to(
        torch.int8
    )
    w2 = (
        (torch.arange(4 * 128 * 64, device="npu").reshape(4, 128, 64) + 11) % 47 - 23
    ).to(torch.int8)
    s1 = torch.full((4, 128, 1), 0.015625, device="npu", dtype=dtype)
    s2 = torch.full_like(s1, 0.015625)
    with torch.no_grad():
        layer.w13_weight.copy_(w1[first : first + 2])
        layer.w2_weight.copy_(w2[first : first + 2])
        layer.w13_weight_scale.copy_(s1[first : first + 2])
        layer.w2_weight_scale.copy_(s2[first : first + 2])
    rows = counts[dp.rank_in_group]
    values = torch.arange(rows * 128, device="npu").reshape(rows, 128)
    x = (((values + dp.rank_in_group * 7) % 29 - 14) * 0.02).to(dtype)
    ids = torch.tensor([[0, 2], [1, 3], [3, 0]], device="npu", dtype=torch.int32)[
        :rows
    ].contiguous()
    probabilities = torch.full((rows, 2), 0.5, device="npu", dtype=torch.float32)
    vector = torch.tensor(counts, dtype=torch.int32, device="cpu")
    original_vector = vector.clone()

    def apply():
        with set_forward_context(
            None, vllm_config, num_tokens=rows, num_tokens_across_dp=vector
        ):
            metadata = get_forward_context().dp_metadata
            assert metadata is not None
            with metadata.sp_local_sizes(1):
                result = method.apply(layer, x, probabilities, ids, None, None)
            assert torch.equal(metadata.num_tokens_across_dp_cpu, original_vector)
        assert torch.equal(vector, original_vector)
        return result

    def oracle():
        return modelslim_w8a8_experts(
            x,
            w1,
            w2,
            s1.float(),
            s2.float(),
            probabilities,
            ids,
            global_num_experts=4,
        )

    with with_strict_mode():
        manager = get_default_manager()
        for op in ("modelslim_w8a8_prepare_experts", "modelslim_w8a8_experts"):
            assert manager.get_selected_impl_id(op) == "vendor.ascend"
        method.process_weights_after_loading(layer)
        assert method.supports_internal_mk
        assert method.moe_kernel.fused_experts.expects_unquantized_inputs
        for _ in range(3):
            eager = apply()
        torch.npu.synchronize()
        assert eager.shape == (rows, 128)
        torch.testing.assert_close(eager, oracle(), rtol=0.03, atol=0.06)
        dist.barrier(group=ep.cpu_group)
        _record("capture_begin", dtype=str(dtype), counts=list(counts), local_rows=rows)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = apply()
        torch.npu.synchronize()
        _record("capture_complete", dtype=str(dtype), counts=list(counts), captures=1)
        for step in range(2):
            x.add_(0.03125)
            expected = oracle()
            graph.replay()
            torch.npu.synchronize()
            assert output.shape == (rows, 128)
            assert output.dtype == dtype
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, rtol=0.03, atol=0.06)
            assert torch.equal(vector, original_vector)
            _record(
                "replay_complete",
                dtype=str(dtype),
                counts=list(counts),
                local_rows=rows,
                replay=step + 1,
                metadata=vector.tolist(),
            )
        dist.barrier(group=ep.cpu_group)
