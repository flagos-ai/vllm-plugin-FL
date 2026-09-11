"""Repro v2: add the two biggest vllm-vs-repro deltas.

E. flaggems: like B, but `import flag_gems` (kernel registration +
   backend activation) happens in each rank before the copy.
F. mmap: like B, but the source is a safetensors file loaded via mmap
   (file-backed pages, as vllm's weight loader does), copied in
   many-medium-tensor chunks like per-weight copies.
"""

import os
import subprocess
import sys
import time

WORLD = 2
PORT = 29512
PER_CONFIG_TIMEOUT = 150

WORKER = r"""
import os, sys, time
rank, world, use_dist, mode = (int(sys.argv[1]), int(sys.argv[2]),
    sys.argv[3] == "1", sys.argv[4])
import torch, torch_musa
if mode in ("E", "F"):
    import flag_gems  # registers aten kernels + backend activation, as in vllm
if use_dist:
    torch.distributed.init_process_group(
        backend="mccl", init_method="tcp://127.0.0.1:%d" % %PORT%,
        world_size=world, rank=rank)
dev = "musa:%d" % rank
torch.musa.set_device(rank)
t0 = time.time()
if mode == "E":
    cpu = torch.randn(4096, 4096).pin_memory()
    gpu = cpu.to(dev, non_blocking=True)
    torch.musa.synchronize()
    print("rank%d single copy OK %.2fs" % (rank, time.time() - t0), flush=True)
elif mode == "F":
    import safetensors.torch
    n, chunk = 256, 4096  # 256 tensors of 4096x4KB-ish rows => ~4MB each
    path = "/tmp/repro_weights_rank%d.safetensors" % rank
    if not os.path.exists(path):
        safetensors.torch.save_file(
            {f"w{i}": torch.randn(1024, 1024) for i in range(n)}, path)
    mm = safetensors.torch.load_file(path)  # mmap-backed
    gpus = {}
    for i in range(n):
        gpus[f"w{i}"] = mm[f"w{i}"].to(dev, non_blocking=True)
    torch.musa.synchronize()
    print("rank%d %d mmap copies (%.0fMB) OK %.2fs" % (
        rank, n, n * 4, time.time() - t0), flush=True)
if use_dist:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
""".replace("%PORT%", str(PORT))


def run_config(name, use_dist, mode, prepare=False):
    logdir = "/tmp/musa_repro"
    os.makedirs(logdir, exist_ok=True)
    procs, ok = [], [False] * WORLD
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
                        mode,
                    ],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            )
    deadline = time.time() + PER_CONFIG_TIMEOUT
    while time.time() < deadline and not all(ok):
        for rank, p in enumerate(procs):
            if p.poll() is not None:
                ok[rank] = p.returncode == 0
        time.sleep(1)
    elapsed = time.time() - t0
    verdict = "PASS" if all(ok) else "HANG"
    print(f"[{name}] {verdict} ({elapsed:.0f}s)")
    for p in procs:
        if p.poll() is None:
            p.kill()
    return verdict


if __name__ == "__main__":
    import torch
    import torch_musa  # noqa: F401

    print("torch_musa devices:", torch.musa.device_count())
    run_config("E_flaggems_mccl", True, "E")
    run_config("F_mmap_mccl", True, "F")
