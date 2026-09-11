"""Minimal multiprocess repro matrix for the MUSA TP>=2 weight-load hang.

Isolates the trigger one factor at a time, without vllm:
  A. dist=False, device=own,  pinned,  non_blocking  -- pure torch_musa multiproc h2d
  B. dist=True,  device=own,  pinned,  non_blocking  -- + mccl context
  C. dist=True,  device=0,    pinned,  non_blocking  -- control: both ranks on dev0
  D. dist=True,  device=own,  no-pin,  blocking      -- drop pin/non_blocking

Each config launches `world` independent python processes (spawn semantics
via subprocess, matching vllm) with a hard timeout. A config "HANGS" when
any rank fails to report within the timeout.
"""

import os
import subprocess
import sys
import time

WORLD = 2
PORT = 29511
PER_CONFIG_TIMEOUT = 150  # seconds; a healthy copy takes <5s

WORKER = r"""
import os, sys, time
rank, world, use_dist, device_mode, pin, nonblocking = (int(sys.argv[1]), int(sys.argv[2]),
    sys.argv[3] == "1", sys.argv[4], sys.argv[5] == "1", sys.argv[6] == "1")
import torch, torch_musa
if use_dist:
    torch.distributed.init_process_group(
        backend="mccl", init_method="tcp://127.0.0.1:%d" % %PORT%,
        world_size=world, rank=rank)
dev = "musa:%d" % rank if device_mode == "own" else "musa:0"
torch.musa.set_device(int(dev.split(":")[1]))
cpu = torch.randn(4096, 4096)
if pin:
    cpu = cpu.pin_memory()
t0 = time.time()
gpu = cpu.to(dev, non_blocking=nonblocking)
torch.musa.synchronize()
print("rank%d copied %.1fMB in %.2fs OK" % (rank, cpu.numel() * 4 / 1e6, time.time() - t0), flush=True)
if use_dist:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
""".replace("%PORT%", str(PORT))


def run_config(name, use_dist, device_mode, pin, nonblocking):
    procs = []
    logdir = "/tmp/musa_repro"
    os.makedirs(logdir, exist_ok=True)
    t0 = time.time()
    for rank in range(WORLD):
        with open(f"{logdir}/{name}_rank{rank}.log", "w") as lf:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        WORKER,
                        str(rank),
                        str(WORLD),
                        str(int(use_dist)),
                        device_mode,
                        str(int(pin)),
                        str(int(nonblocking)),
                    ],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            )
    deadline = time.time() + PER_CONFIG_TIMEOUT
    ok = [False] * WORLD
    while time.time() < deadline and not all(ok):
        for rank, p in enumerate(procs):
            if p.poll() is not None:
                ok[rank] = p.returncode == 0
        time.sleep(1)
    elapsed = time.time() - t0
    if all(ok):
        print(f"[{name}] PASS ({elapsed:.0f}s)")
    else:
        print(
            f"[{name}] HANG -> killed after {PER_CONFIG_TIMEOUT}s; "
            f"rank logs: {logdir}/{name}_rank*.log"
        )
    for p in procs:
        if p.poll() is None:
            p.kill()
        elif p.returncode not in (0, None):
            ok = [False] * WORLD
    return all(ok)


if __name__ == "__main__":
    import torch
    import torch_musa  # noqa: F401

    print("torch_musa devices:", torch.musa.device_count())
    configs = [
        ("A_pure_multiproc", False, "own", True, True),
        ("B_mccl_own_dev", True, "own", True, True),
        ("C_mccl_all_dev0", True, "0", True, True),
        ("D_mccl_nopin_block", True, "own", False, False),
    ]
    for name, use_dist, dev_mode, pin, nb in configs:
        run_config(name, use_dist, dev_mode, pin, nb)
