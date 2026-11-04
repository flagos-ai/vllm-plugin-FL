"""Bounded BF16 GLM payload probe; invoked by dual_node_smoke collective-bench.

Uses the same PyNcclCommunicator as the model, without model allocations or
additional torch NCCL groups. algbw is payload/time, NOT physical NIC bandwidth.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def counters():
    paths = list(Path("/sys/class/infiniband").glob("*/ports/1/counters/port_*_data"))
    paths += list(Path("/sys/class/net/bond1/statistics").glob("*_bytes"))
    paths += list(Path("/sys/kernel/mm/memory_peers/gpukfd").glob("num_*"))
    result = {}
    for path in paths:
        try:
            result[str(path)] = int(path.read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def worker(local_rank, node_rank, local_size, world_size, master, port, tokens, iters):
    import torch
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    rank = node_rank * local_size + local_rank
    torch.cuda.set_device(local_rank)
    group = StatelessProcessGroup.create(master, port, rank, world_size, store_timeout=120)
    before = counters() if local_rank == 0 else {}
    comm = PyNcclCommunicator(group, local_rank)
    assert comm.available and not comm.disabled
    maps = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if any(name in line for name in ("librccl", "libibverbs", "libshca", "libucp", "libucs", "libuct", "libucm"))})
    print("COLLECTIVE_IDENTITY", json.dumps({"rank": rank, "pid": os.getpid(),
          "torch": torch.__version__, "hip": torch.version.hip,
          "nccl_version": comm.nccl_version, "libraries": {
              name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in maps},
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for token_count in map(int, tokens.split(",")):
            count = token_count * 6144
            # Integers keep EVERY intermediate sum exact in BF16 for <=16 ranks.
            # Fractional inputs can round inside the reduction tree even when
            # inputs and the final mathematical sum are exactly representable.
            pattern = (torch.arange(count, device="cuda", dtype=torch.int32) % 4).to(torch.bfloat16)
            x = pattern + rank + 1
            out = torch.empty_like(x)
            expected = pattern * world_size + world_size * (world_size + 1) / 2
            comm.all_reduce(x, out, stream=stream)
            stream.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            for mode in (["eager", "graph"] if token_count <= 64 else ["eager"]):
                for _ in range(3):
                    comm.all_reduce(x, out, stream=stream)
                stream.synchronize()
                graph = None
                if mode == "graph":
                    group.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        comm.all_reduce(x, out, stream=stream)
                    x.add_(1)
                    graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(out, expected + world_size, rtol=0, atol=0)
                    x.sub_(1)
                    for _ in range(3):
                        graph.replay()
                    stream.synchronize()
                rows = []
                for repeat in range(3):
                    group.barrier()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    stream.synchronize()
                    wall = time.perf_counter()
                    begin.record(stream)
                    for _ in range(iters):
                        if graph is not None:
                            graph.replay()
                        else:
                            comm.all_reduce(x, out, stream=stream)
                    end.record(stream)
                    end.synchronize()
                    rows.append({"repeat": repeat, "event_ms": begin.elapsed_time(end) / iters,
                                 "wall_ms": (time.perf_counter() - wall) * 1000 / iters})
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                rank_rows = group.all_gather_obj({"rank": rank, "rows": rows})
                if rank == 0:
                    maxima = [{"repeat": i, **{key: max(item["rows"][i][key] for item in rank_rows)
                              for key in ("event_ms", "wall_ms")}} for i in range(3)]
                    print("COLLECTIVE_RESULT", json.dumps({"tokens": token_count, "bytes": count * 2,
                          "dtype": "bfloat16", "world_size": world_size, "mode": mode,
                          "iters": iters, "correctness": "PASS", "max_rank": maxima,
                          "ranks": rank_rows}), flush=True)
                if graph is not None:
                    graph.reset()
                    del graph
                    gc.collect()
                group.barrier()
            del x, out, pattern, expected
    torch.cuda.synchronize()
    group.barrier()
    # Graphs are released before collective destruction, unlike the old smoke.
    comm.nccl.ncclCommDestroy(comm.comm)
    comm.disabled, comm.available = True, False
    group.barrier()
    print(f"COLLECTIVE_CLEANUP_PASS rank={rank}", flush=True)
    if local_rank == 0:
        after = counters()
        print("COLLECTIVE_COUNTERS", json.dumps({"rank": rank, "before": before,
              "after": after, "delta": {p: after[p] - v for p, v in before.items() if p in after},
              "scope": "host interface counters; may include unrelated host traffic"}), flush=True)
