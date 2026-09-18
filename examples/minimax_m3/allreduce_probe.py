# SPDX-License-Identifier: Apache-2.0
"""TP16 AllReduce on native torch/MCCL, with numerical and HCA-counter evidence."""

import argparse
import datetime
import gc
import json
import os
import statistics
import time
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--mode", choices=["eager", "graph"], default="eager")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--tokens", default="3,64,401,1024,8192")
ap.add_argument("--output-dir", type=Path, default=Path("allreduce-results"))
a = ap.parse_args()
rank = int(os.environ["RANK"])
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
root = a.output_dir / a.case
root.mkdir(parents=True, exist_ok=True)
log = (root / f"rank{rank}.jsonl").open("w", buffering=1)


def emit(stage, **kw):
    row = dict(time=time.time(), rank=rank, stage=stage, **kw)
    log.write(json.dumps(row) + "\n")
    if rank in (0, 8):
        print(json.dumps(row), flush=True)


def counters():
    out = {}
    for dev in sorted(Path("/sys/class/infiniband").glob("*")):
        row = {}
        for directory in ("hw_counters", "counters"):
            for p in (dev / "ports/1" / directory).glob("*"):
                if p.name in ("lifespan",):
                    continue
                with suppress(ValueError, OSError):
                    row[directory + "/" + p.name] = int(p.read_text().strip())
        out[dev.name] = row
    return out


emit(
    "init",
    case=a.case,
    mode=a.mode,
    env={k: v for k, v in os.environ.items() if k.startswith(("MCCL_", "NCCL_"))},
    torch=torch.__version__,
)
dist.init_process_group(
    "nccl",
    timeout=datetime.timedelta(seconds=120),
    device_id=torch.device("cuda", local),
)
cpu = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=150))
emit("connected", world=dist.get_world_size())
assert dist.get_world_size() == 16
dist.barrier(group=cpu)
if local == 0:
    emit("idle_before", counters=counters())
time.sleep(1)
if local == 0:
    emit("idle_after", counters=counters())
dist.barrier(group=cpu)
results = []
for tokens in map(int, a.tokens.split(",")):
    x = torch.empty((tokens, 6144), device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        x.fill_(rank + 1)
        dist.all_reduce(x)
    torch.cuda.synchronize()
    graph = None
    if a.mode == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.fill_(rank + 1)
            dist.all_reduce(x)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dist.all_reduce(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
    dist.barrier(group=cpu)
    before = counters() if local == 0 else None
    if local == 0:
        emit(
            "before", tokens=tokens, bytes=x.numel() * x.element_size(), counters=before
        )
    dist.barrier(group=cpu)
    samples = []
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(a.warmup + a.iters):
        x.fill_(rank + 1)
        start.record()
        if graph is None:
            dist.all_reduce(x)
        else:
            graph.replay()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if i >= a.warmup:
            samples.append(ms)
        if i == 0:
            emit("first_iteration", tokens=tokens, ms=ms)
    assert torch.equal(x, torch.full_like(x, 136)), ("allreduce mismatch", rank, tokens)
    # Every possible intermediate sum is exactly representable in BF16.
    index = torch.arange(x.numel(), device="cuda", dtype=torch.int32)
    pattern = (
        (((index ^ (index >> 7) ^ (index >> 17)) & 3) - 1).to(torch.bfloat16).view_as(x)
    )
    x.copy_(pattern * (rank + 1))
    if graph is None:
        dist.all_reduce(x)
    else:
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, pattern * 136), ("pattern allreduce mismatch", rank, tokens)
    del pattern, index
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    if local == 0:
        after = counters()
        delta = {
            h: {k: v - before[h].get(k, 0) for k, v in row.items()}
            for h, row in after.items()
        }
        emit("after", tokens=tokens, counters=after, delta=delta)
    row = dict(
        tokens=tokens,
        bytes=x.numel() * x.element_size(),
        mode=a.mode,
        iters=a.iters,
        warmup=a.warmup,
        mean_ms=statistics.mean(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        samples_ms=samples,
        correct=True,
        pattern_correct=True,
        wall_s=time.perf_counter() - wall_start,
    )
    results.append(row)
    emit("case_pass", **{k: v for k, v in row.items() if k != "samples_ms"})
    del graph, x, start, end
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
(root / f"rank{rank}.json").write_text(
    json.dumps(dict(rank=rank, case=a.case, all_pass=True, rows=results), indent=2)
)
emit("all_pass")
dist.barrier(group=cpu)
dist.destroy_process_group(cpu)
dist.destroy_process_group()
emit("clean_exit")
