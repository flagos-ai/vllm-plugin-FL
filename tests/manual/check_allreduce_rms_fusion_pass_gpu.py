# SPDX-License-Identifier: Apache-2.0
"""Exercise real vLLM graph matching and the FL op, without loading weights.

This is correctness validation, not a model throughput benchmark. Launch with
torchrun and two GPUs; full model inference is a separate required check.
"""

import json
import os
from pathlib import Path


def main():
    import torch
    import torch._inductor.pattern_matcher as pm
    import torch.distributed as dist
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm.ir.ops
    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import PassConfig
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )

    from vllm_fl.patches.allreduce_rms_provider import (
        allreduce_rms_provider_stats,
        apply_allreduce_rms_provider_patch,
    )

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.set_grad_enabled(False)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    config = VllmConfig(
        model_config=ModelConfig(
            model="/models/chenzhuo/Qwen3.6-27B", max_model_len=8192
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=257,
            max_num_seqs=8,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            pass_config=PassConfig(fuse_allreduce_rms=True)
        ),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local,
            distributed_init_method="env://",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
        assert apply_allreduce_rms_provider_patch()
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            AllReduceFusionPass,
        )

        fusion = AllReduceFusionPass(config)
        hidden = config.model_config.get_hidden_size()

        def fn(x, residual, gamma):
            reduced = tensor_model_parallel_all_reduce(x)
            return vllm.ir.ops.fused_add_rms_norm(
                reduced, residual, gamma.float() + 1.0, 1e-6
            )

        with FakeTensorMode():
            inputs = [
                torch.empty((32, hidden), device=device, dtype=torch.bfloat16),
                torch.empty((32, hidden), device=device, dtype=torch.bfloat16),
                torch.empty(hidden, device=device, dtype=torch.bfloat16),
            ]
            gm = pm.fwd_only(fn, inputs)
            fusion(gm.graph)
            gm.recompile()
        graph_path = Path(f"/results/allreduce_pass_gpu_graph_rank{rank}.py")
        graph_path.write_text(gm.code)
        assert "fl_pr773_fused_allreduce_norm" in gm.code, gm.code
        assert "flashinfer_trtllm_fused_allreduce_norm" not in gm.code
        torch.manual_seed(179 + rank)
        real_inputs = [
            torch.randn((32, hidden), device=device, dtype=torch.bfloat16) * 0.2,
            torch.randn((32, hidden), device=device, dtype=torch.bfloat16) * 0.1,
            torch.full((hidden,), 0.25, device=device, dtype=torch.bfloat16),
        ]
        # Mirror actual vLLM model weights: Parameter flags remain True under
        # no_grad, even though this invocation does not request autograd.
        real_inputs[2] = torch.nn.Parameter(real_inputs[2])
        gm(*real_inputs)
        torch.cuda.synchronize()
        dist.barrier(group=get_tp_group().cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = gm(*real_inputs)
        torch.cuda.synchronize()
        dist.barrier(group=get_tp_group().cpu_group)
        for iteration in range(3):
            source = torch.randn_like(real_inputs[0]) * 0.2
            residual = torch.randn_like(real_inputs[1]) * 0.1
            summed = source.float()
            dist.all_reduce(summed)
            expected_residual = (summed.to(source.dtype).float() + residual.float()).to(
                source.dtype
            )
            value = expected_residual.float()
            expected_norm = (
                value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * 1.25
            ).to(source.dtype)
            real_inputs[0].copy_(source)
            real_inputs[1].copy_(residual)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output[0], expected_norm, atol=0.04, rtol=0.04)
            torch.testing.assert_close(
                output[1], expected_residual, atol=0.04, rtol=0.04
            )
        stats = allreduce_rms_provider_stats()
        assert stats["host_dispatches"].get("peer:pattern=1", 0) > 0, stats
        assert stats["host_dispatches"].get("native", 0) == 0, stats
        print(
            json.dumps(
                dict(
                    rank=rank,
                    status="passed",
                    real_vllm_pass=True,
                    graph_replays=3,
                    hidden=hidden,
                    stats=stats,
                )
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
