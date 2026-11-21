# SPDX-License-Identifier: Apache-2.0
"""GPU correctness only (no timings, no alternate model benchmark).

torchrun --standalone --nproc_per_node=2 \
  tests/manual/check_allreduce_rms_provider_gpu.py --backend peer

Use 8 ranks for TP8; --tp-size 2 under 4/8 ranks exercises independent TP
subgroups rather than WORLD. Every rank emits JSON only after all checks pass.
This does not establish actual model graph fusion: separately require a real
FL model run to log `PR773 first host dispatch` and check its generated output.
"""

import argparse
import json
import os


def main():
    import torch
    import torch.distributed as dist

    from vllm_fl.patches import _allreduce_rms_pr773 as provider

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("peer", "mnnvl"), required=True)
    parser.add_argument("--tp-size", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--weight-bias", type=float, default=0.0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 65, 257])
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    tp = args.tp_size or world
    assert tp in (2, 4, 8) and world % tp == 0
    cpu_group = reference_group = None
    for start in range(0, world, tp):
        ranks = list(range(start, start + tp))
        cpu = dist.new_group(ranks, backend="gloo")
        nccl = dist.new_group(ranks, backend="nccl")
        if rank in ranks:
            cpu_group, reference_group = cpu, nccl
    torch.manual_seed(71 + rank)
    workspace = provider.create_fused_allreduce_rms_norm_workspace(
        world_size=tp,
        rank=dist.get_rank(cpu_group),
        max_token_num=max(args.tokens),
        hidden_dim=args.hidden,
        dtype=torch.bfloat16,
        group=cpu_group,
        backend=args.backend,
    )
    cases = []
    for tokens in args.tokens:
        for pattern in (1, 2):
            for explicit_norm in (False, True):
                x = torch.empty(
                    (tokens, args.hidden), device=device, dtype=torch.bfloat16
                )
                residual = torch.empty_like(x)
                gamma = torch.full((args.hidden,), 0.25, device=device, dtype=x.dtype)
                norm_out = torch.empty_like(x) if explicit_norm else None
                quant_out = (
                    torch.empty_like(x, dtype=torch.float8_e4m3fn)
                    if pattern == 2
                    else None
                )
                scale = (
                    torch.tensor([1.25], dtype=torch.float32, device=device)
                    if pattern == 2
                    else None
                )
                kwargs = dict(
                    allreduce_in=x,
                    residual=residual,
                    rms_gamma=gamma,
                    rms_eps=1e-6,
                    world_size=tp,
                    launch_with_pdl=True,
                    fp32_acc=True,
                    max_token_num=max(args.tokens),
                    pattern_code=pattern,
                    norm_out=norm_out,
                    quant_out=quant_out,
                    scale_factor=scale,
                    workspace=workspace,
                    weight_bias=args.weight_bias,
                )
                # Initialize payload before eager warmup and capture. Every
                # replay below replaces it with fresh values on every rank.
                x.normal_(std=0.2)
                residual.normal_(std=0.1)
                provider.fused_allreduce_rms_norm(**kwargs)
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    provider.fused_allreduce_rms_norm(**kwargs)
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)
                cases.append((kwargs, graph))
    checked = 0
    # Interleave captured shapes and quant patterns sharing one stateful
    # workspace. Reuse of capture-time values cannot satisfy these checks.
    for iteration in range(3):
        for kwargs, graph in cases:
            x, residual = kwargs["allreduce_in"], kwargs["residual"]
            source = torch.randn_like(x) * 0.2
            old_residual = torch.randn_like(x) * 0.1
            summed = source.float()
            dist.all_reduce(summed, group=reference_group)
            expected_residual = (summed.to(x.dtype).float() + old_residual.float()).to(
                x.dtype
            )
            value = expected_residual.float()
            expected_norm = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * (kwargs["rms_gamma"].float() + args.weight_bias)
            ).to(x.dtype)
            x.copy_(source)
            residual.copy_(old_residual)
            if kwargs["norm_out"] is not None:
                kwargs["norm_out"].fill_(7)
            graph.replay()
            torch.cuda.synchronize()
            explicit_norm = kwargs["norm_out"] is not None
            actual_residual = x if explicit_norm else residual
            torch.testing.assert_close(
                actual_residual, expected_residual, atol=0.04, rtol=0.04
            )
            if explicit_norm:
                torch.testing.assert_close(residual, old_residual, atol=0, rtol=0)
            if kwargs["pattern_code"] == 1:
                actual_norm = kwargs["norm_out"] if explicit_norm else x
                torch.testing.assert_close(
                    actual_norm, expected_norm, atol=0.04, rtol=0.04
                )
            else:
                expected_quant = (
                    (expected_norm.float() / 1.25)
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                actual_bits = kwargs["quant_out"].view(torch.uint8)
                expected_bits = expected_quant.view(torch.uint8)
                actual_mag = (actual_bits & 127).to(torch.int16)
                expected_mag = (expected_bits & 127).to(torch.int16)
                same_sign = (actual_bits & 128) == (expected_bits & 128)
                adjacent = same_sign & ((actual_mag - expected_mag).abs() <= 1)
                both_zero = (actual_mag == 0) & (expected_mag == 0)
                assert bool(torch.all(adjacent | both_zero).item()), (
                    "FP8 differs by more than one adjacent code"
                )
                if explicit_norm:
                    torch.testing.assert_close(
                        kwargs["norm_out"], torch.full_like(x, 7), atol=0, rtol=0
                    )
                else:
                    torch.testing.assert_close(x, source, atol=0, rtol=0)
            checked += 1
    dist.barrier(group=cpu_group)
    print(
        json.dumps(
            dict(
                rank=rank,
                tp_rank=dist.get_rank(cpu_group),
                tp_size=tp,
                backend=args.backend,
                hidden=args.hidden,
                checks=checked,
                weight_bias=args.weight_bias,
                graph_interleaving=True,
                patterns=[1, 2],
                status="passed",
            )
        ),
        flush=True,
    )
    workspace.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
