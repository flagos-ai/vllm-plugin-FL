# Copyright (c) 2026 BAAI. All rights reserved.
"""Check real FlagCX TXDA tensor communication before a model acceptance run."""

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path


def _worker(rank, world_size, port, results):
    report = {"rank": rank, "passed": []}
    comm = None
    dist = None
    try:
        import torch
        import torch.distributed as dist
        import torch_txda  # noqa: F401

        import vllm_fl.distributed.device_communicators.flagcx as api

        torch.set_num_threads(1)
        torch.txda.set_device(rank)
        device = torch.device("txda", rank)
        library = api._resolve_flagcx_library_path()
        report.update(
            library=library,
            library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            wrapper=api.FLAGCXLibrary.__module__,
        )
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120),
        )
        comm = api.PyFlagcxCommunicator(dist.group.WORLD, device)
        if not comm.available or comm.disabled:
            raise RuntimeError("FlagCX did not initialize") from comm.init_error
        if not comm._sync_collectives:
            raise RuntimeError("This probe requires VLLM_TXDA_SYNC_COLLECTIVES=1")
        report["passed"].append("initialization")
        total = world_size * (world_size + 1) / 2

        def check(name, actual, expected):
            if actual != expected:
                raise AssertionError(
                    f"{name}: actual={actual!r}, expected={expected!r}"
                )
            dist.barrier()
            report["passed"].append(name)
            print(f"rank={rank} PASS {name}", flush=True)

        for dtype in (torch.float32, torch.bfloat16):
            for count in (1, 4, 64):
                value = torch.full((count,), rank + 1, device=device, dtype=dtype)
                output = comm.all_reduce(value)
                check(
                    f"all_reduce/{dtype}/{count}",
                    output.cpu().tolist(),
                    [total] * count,
                )
        for dtype in (torch.int32, torch.int64):
            value = torch.full((4,), rank + 1, device=device, dtype=dtype)
            output = torch.empty(4 * world_size, device=device, dtype=dtype)
            comm.all_gather(output, value)
            check(
                f"all_gather/{dtype}",
                output.cpu().tolist(),
                [peer + 1 for peer in range(world_size) for _ in range(4)],
            )
        sizes = list(range(1, world_size + 1))
        value = torch.full((sizes[rank],), rank + 1, device=device, dtype=torch.int32)
        output = torch.empty(sum(sizes), device=device, dtype=torch.int32)
        comm.all_gatherv(output, value, sizes)
        check(
            "all_gatherv",
            output.cpu().tolist(),
            [peer + 1 for peer in range(world_size) for _ in range(sizes[peer])],
        )
        value = torch.full((4 * world_size,), rank + 1, device=device)
        output = torch.empty(4, device=device)
        comm.reduce_scatter(output, value)
        check("reduce_scatter", output.cpu().tolist(), [total] * 4)
        value = torch.full((sum(sizes),), rank + 1, device=device)
        output = torch.empty(sizes[rank], device=device)
        comm.reduce_scatterv(output, value, sizes)
        check("reduce_scatterv", output.cpu().tolist(), [total] * sizes[rank])
        value = torch.full((4,), 7 if rank == 0 else 0, device=device)
        comm.broadcast(value, 0)
        check("broadcast", value.cpu().tolist(), [7.0] * 4)
        value = torch.full((64,), rank + 1, device=device)
        comm.group_start()
        comm.group_start()
        first = comm.all_reduce(value)
        second = comm.all_reduce(value)
        comm.group_end()
        comm.group_end()
        check("nested_group/first", first.cpu().tolist(), [total] * 64)
        check("nested_group/second", second.cpu().tolist(), [total] * 64)
        report["ok"] = True
    except BaseException as error:
        report.update(ok=False, error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        if comm is not None:
            comm.destroy()
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        results.put(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=29519)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("FlagCX acceptance requires at least two devices")
    if os.getenv("VLLM_TXDA_SYNC_COLLECTIVES") != "1":
        parser.error("source the TXDA environment with VLLM_TXDA_SYNC_COLLECTIVES=1")
    context = mp.get_context("spawn")
    results = context.Queue()
    children = [
        context.Process(
            target=_worker, args=(rank, args.world_size, args.port, results)
        )
        for rank in range(args.world_size)
    ]
    for process in children:
        process.start()
    deadline = time.monotonic() + args.timeout
    reports = []
    for _ in children:
        try:
            reports.append(results.get(timeout=max(0.1, deadline - time.monotonic())))
        except queue.Empty:
            break
    for process in children:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in children:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    report = {
        "ok": len(reports) == args.world_size
        and all(item["ok"] for item in reports)
        and all(process.exitcode == 0 for process in children),
        "world_size": args.world_size,
        "exit_codes": [process.exitcode for process in children],
        "ranks": sorted(reports, key=lambda item: item["rank"]),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
