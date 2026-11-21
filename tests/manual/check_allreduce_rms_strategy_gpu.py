# SPDX-License-Identifier: Apache-2.0
"""Check partial-prefill strategy transitions and changed-input graph replay.

Run from the repository on two available NVIDIA GPUs:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/manual/check_allreduce_rms_strategy_gpu.py

This checks protocol retirement and sampled-row numerical correctness. It is
not a model throughput benchmark. The sequence covers peer TP2/BF16,
PDL on/off, partial two-shot prefills and both one-shot implementations.
"""

import json
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    from vllm_fl.patches import _allreduce_rms_pr773 as mod

    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")
    torch.set_grad_enabled(False)
    H = 5120
    M = 4096
    dtype = torch.bfloat16
    w = mod.create_fused_allreduce_rms_norm_workspace(
        world_size=2,
        rank=rank,
        max_token_num=M,
        hidden_dim=H,
        dtype=dtype,
        group=cpu,
        backend="peer",
    )
    records = []
    gamma = torch.full((H,), 0.25, device=device, dtype=dtype)

    def note(d):
        d.update(rank=rank)
        records.append(d)
        print(json.dumps(d), flush=True)

    def reset():
        mod._peer_initialize_sentinel_workspace_kernel[
            ((w.local_buffer.numel() + 255) // 256,)
        ](w.local_buffer, w.state, w.local_buffer.numel(), BLOCK_SIZE=256, num_warps=4)
        torch.cuda.synchronize()
        dist.barrier(group=cpu)

    def call(x, r, pdl=False):
        return mod.fused_allreduce_rms_norm(
            x, r, gamma, 1e-6, 2, pdl, True, M, 1, weight_bias=1.0, workspace=w
        )

    for following in [64, 1]:
        reset()
        for rows in [3500, following]:
            x = torch.full((rows, H), 0.1 * (rank + 1), device=device, dtype=dtype)
            r = torch.full_like(x, 0.25)
            call(x, r)
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
        state = w.state.cpu().tolist()
        dirty = int(
            (w.local_buffer.view(-1)[: 2 * M * H].view(torch.int16) != -32768)
            .sum()
            .item()
        )
        expected = 0
        note(
            {
                "stage": "clear_layout_probe",
                "previous_rows": 3500,
                "following_rows": following,
                "state": state,
                "dirty_elements_after_retirement": dirty,
                "expected_dirty": expected,
                "sample_at_first_dirty_location": int(
                    w.local_buffer.view(-1).view(torch.int16)[3500 * H].item()
                ),
            }
        )
        assert dirty == expected, (dirty, expected)
        assert state[3] == 0, state
    seq = [3500, 64, 3751, 34, 4096, 96, 3500, 1, 3751, 128, 4096, 25, 3501, 64, 2]
    tensors = {
        m: (
            torch.empty((m, H), device=device, dtype=dtype),
            torch.empty((m, H), device=device, dtype=dtype),
        )
        for m in set(seq)
    }
    base = [
        ((torch.arange(H) % 17 + 1).float() * (0.03 * (i + 1))).to(dtype)
        for i in range(2)
    ]
    graphs = {}
    for pdl in [False, True]:
        reset()
        for rows, (x, r) in tensors.items():
            x.copy_(base[rank].expand(rows, -1))
            r.fill_(0.25)
            call(x, r, pdl)
            torch.cuda.synchronize()
            if rows in [1, 34, 64, 96, 128]:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call(x, r, pdl)
                graphs[rows] = graph
        for iteration in range(12):
            phase = 0.001 * iteration
            a = (base[0].float() + phase).to(dtype)
            b = (base[1].float() + phase).to(dtype)
            ref_residual = ((a.float() + b.float()).to(dtype).float() + 0.25).to(dtype)
            ref_norm = (
                ref_residual.float()
                * torch.rsqrt(ref_residual.float().square().mean() + 1e-6)
                * 1.25
            ).to(dtype)
            for rows in seq:
                x, r = tensors[rows]
                x.copy_((base[rank].float() + phase).to(dtype).expand(rows, -1))
                r.fill_(0.25)
                if rows in graphs:
                    graphs[rows].replay()
                else:
                    call(x, r, pdl)
                torch.cuda.synchronize()
                samples = torch.linspace(
                    0, rows - 1, min(rows, 128), device=device
                ).long()
                torch.testing.assert_close(
                    x[samples].cpu(),
                    ref_norm.expand(len(samples), -1),
                    rtol=0.02,
                    atol=0.02,
                )
                torch.testing.assert_close(
                    r[samples].cpu(),
                    ref_residual.expand(len(samples), -1),
                    rtol=0.02,
                    atol=0.02,
                )
            note(
                {
                    "stage": "mixed_shape_numerical_and_changed_input_graph_passed",
                    "pdl": pdl,
                    "iteration": iteration,
                    "sequence": seq,
                    "sampled_rows_per_call": 128,
                }
            )
    dist.barrier(group=cpu)
    w.destroy()
    dist.destroy_process_group()
    print(
        json.dumps({"status": "passed", "rank": rank, "records": records}), flush=True
    )


if __name__ == "__main__":
    main()
