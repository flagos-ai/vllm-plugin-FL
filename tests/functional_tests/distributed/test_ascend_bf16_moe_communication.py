# SPDX-License-Identifier: Apache-2.0
"""NPU-only component gate, launched with torchrun (no model required).

Example: torchrun --nproc-per-node=8 <this-file> --mode alltoall --hidden 6144
Run mc2 separately with --mode mc2. Cross-host launch uses the same CLI and
normal torchrun rendezvous arguments. Every rank must enter both collectives.
"""

import argparse
import os

import torch
import torch.distributed as dist


def main():
    import torch_npu

    from vllm_fl.dispatch import CachedOp
    from vllm_fl.distributed.device_communicators.npu_moe import (
        NpuAllToAllDispatcher,
        NpuMC2Dispatcher,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mc2", "alltoall"), required=True)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=16)
    args = parser.parse_args()
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.experts % world == 0 and world > 1
    device = torch.device("npu", torch_npu.npu.current_device())
    dtype = torch.bfloat16
    # Rank zero exercises idle participation; remaining ranks differ in activity.
    active = 0 if rank == 0 else max(1, args.rows - rank % args.rows)
    rows = args.rows if args.mode == "mc2" else active
    generator = torch.Generator(device="cpu").manual_seed(73 + rank)
    x = (torch.randn(rows, args.hidden, generator=generator) * 0.05).to(device, dtype)
    ids = (
        (torch.arange(rows * args.topk).reshape(rows, args.topk) + rank * args.topk)
        % args.experts
    ).to(device, torch.int32)
    probs = torch.full((rows, args.topk), 1 / args.topk, device=device)
    mask = torch.arange(rows, device=device) < active
    probs = probs * mask[:, None]
    generator.manual_seed(97)
    w1 = (
        torch.randn(
            args.experts, 2 * args.intermediate, args.hidden, generator=generator
        )
        * 0.02
    ).to(device, dtype)
    w2 = (
        torch.randn(args.experts, args.hidden, args.intermediate, generator=generator)
        * 0.02
    ).to(device, dtype)
    local = args.experts // world
    group = dist.new_group(backend="hccl")
    dispatcher = (NpuMC2Dispatcher if args.mode == "mc2" else NpuAllToAllDispatcher)(
        group,
        args.experts,
    )
    routed = (
        dispatcher.dispatch(x, ids, probs, mask)
        if args.mode == "mc2"
        else dispatcher.dispatch(x, ids, probs)
    )
    assert routed.hidden_states.dtype == dtype and routed.pertoken_scale is None
    local_out = CachedOp("bf16_grouped_experts")(
        routed.hidden_states,
        w1[rank * local : (rank + 1) * local],
        w2[rank * local : (rank + 1) * local],
        group_list=routed.group_list,
        group_list_type=routed.group_list_type,
        activation="swigluoai_uninterleave",
        alpha=1.702,
        beta=1.0,
        limit=7.0,
    )
    actual = dispatcher.combine(local_out, routed.handle)
    # Independent explicit reference: packed halves, upper-only gate clamp,
    # symmetric up clamp, alpha sigmoid, beta, routing exactly once.
    expected = torch.zeros_like(x)
    for row in range(active):
        for slot in range(args.topk):
            expert = int(ids[row, slot].item())
            projected = x[row] @ w1[expert].T
            gate, up = projected.chunk(2)
            gate = gate.clamp(max=7)
            up = up.clamp(-7, 7)
            activated = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            expected[row] += (activated @ w2[expert].T) * probs[row, slot]
    torch.testing.assert_close(actual[:active], expected[:active], atol=0.02, rtol=0.05)
    assert torch.isfinite(actual[:active]).all()
    print(
        f"rank={rank} mode={args.mode} active={active} hidden={args.hidden} PASS",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
