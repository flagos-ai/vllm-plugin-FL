# SPDX-License-Identifier: Apache-2.0
"""Bounded controlled warp-skew counterexample for scalar arrival retirement.

This models the observed state=0 versus expected64 loop. The old path has a
bounded spin so it reports missing arrivals without leaving a GPU hung. It is
not an AllReduce numerical qualification or model throughput benchmark.
"""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def retire(
    state, out, salt, GRID: tl.constexpr, LANES: tl.constexpr, FIX: tl.constexpr
):
    pid = tl.program_id(0)
    tl.atomic_add(state, 1, sem="release", scope="gpu")
    if pid == 0:
        # Explicit warp skew represents the scheduler delay between sibling
        # warps observed in the stalled CTA. It preserves every payload value.
        delay = tl.inline_asm_elementwise(
            "{ .reg .u32 t; .reg .pred p; mov.u32 t, %tid.x; setp.ge.u32 p,t,32; @p nanosleep.u32 1000000; mov.u32 $0,0; }",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        arrived = tl.load(state, volatile=True) + delay
        spins = 0
        while (arrived != GRID) & (spins < 100000):
            arrived = tl.load(state, volatile=True)
            spins += 1
        if FIX:
            tl.debug_barrier()
        lanes = tl.arange(0, LANES)
        tl.store(out + lanes, arrived + tl.load(salt))
        tl.atomic_xchg(state, 0, sem="release", scope="gpu")


def main():
    torch.cuda.set_device(0)
    records = []
    for grid, warps in [(32, 4), (64, 8), (132, 32)]:
        lanes = warps * 32
        for fixed in [False, True]:
            state = torch.zeros((), device="cuda", dtype=torch.int32)
            out = torch.empty(lanes, device="cuda", dtype=torch.int32)
            salt = torch.zeros((), device="cuda", dtype=torch.int32)
            values = []
            for v in [0, 7, 19]:
                state.zero_()
                salt.fill_(v)
                retire[(grid,)](state, out, salt, grid, lanes, fixed, num_warps=warps)
                torch.cuda.synchronize()
                observed = out.cpu()
                bad = int((observed != grid + v).sum())
                values.append(
                    {
                        "stage": "eager",
                        "salt": v,
                        "mismatched_threads": bad,
                        "observed_values": sorted(set(observed.tolist())),
                    }
                )
                if fixed:
                    assert bad == 0, values[-1]
            state.zero_()
            salt.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                retire[(grid,)](state, out, salt, grid, lanes, fixed, num_warps=warps)
            for v in [3, 11, 23]:
                state.zero_()
                salt.fill_(v)
                g.replay()
                torch.cuda.synchronize()
                observed = out.cpu()
                bad = int((observed != grid + v).sum())
                values.append(
                    {
                        "stage": "changed_input_graph",
                        "salt": v,
                        "mismatched_threads": bad,
                        "observed_values": sorted(set(observed.tolist())),
                    }
                )
                if fixed:
                    assert bad == 0, values[-1]
            record = {
                "grid": grid,
                "num_warps": warps,
                "fixed": fixed,
                "values": values,
                "scope": "Bounded protocol counterexample; not full AllReduce numerical or throughput",
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    old = [x for x in records if not x["fixed"]]
    assert all(any(v["mismatched_threads"] > 0 for v in x["values"]) for x in old), (
        "Controlled skew did not reproduce every old config"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "old_configs_expose_race": len(old),
                "fixed_configs": 3,
                "fixed_eager_calls": 9,
                "fixed_changed_input_replays": 9,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
