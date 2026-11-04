#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Check the product GDN index mode against whole-allocation CPU oracles."""

import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "cases": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import torch
        import torch_txda  # noqa: F401
        from vllm_fl.worker.txda_gdn_state import TxdaStateCacheIndexMode

        torch.set_num_threads(32)
        torch.txda.set_device(0)
        device = torch.device("txda:0")
        generator = torch.Generator().manual_seed(20261104)
        shape = (20, 128, 128)
        row_elements = 20 * 128 * 128
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                for blocks in (32, 167):
                    stride = row_elements + 65536
                    total = 11 + blocks * stride + 13
                    initial = torch.randint(
                        -125, 126, (total,), dtype=torch.int16, generator=generator
                    ).to(dtype)
                    backing = initial.to(device)
                    cache = backing.as_strided(
                        (blocks, *shape), (stride, 16384, 128, 1), storage_offset=11
                    )
                    cpu_cache = initial.as_strided(
                        cache.shape, cache.stride(), cache.storage_offset()
                    )
                    for count in (1, 8):
                        ids = [7] if count == 1 else [7, 3, -1, 0, 12, 1, 17, 6]
                        index_cpu = torch.tensor(ids, dtype=torch.int32)
                        index = index_cpu.to(device)
                        values_cpu = (
                            torch.randn((count, *shape), generator=generator) * 0.25
                        ).to(dtype)
                        values = values_cpu.to(device)
                        expected_backing = initial.clone()
                        expected_cache = expected_backing.as_strided(
                            cache.shape, cache.stride(), cache.storage_offset()
                        )
                        expected_gather = cpu_cache[index_cpu]
                        expected_cache[index_cpu] = values_cpu
                        backing.copy_(initial)
                        torch.txda.current_stream().synchronize()
                        mode = TxdaStateCacheIndexMode(cache)
                        started = time.perf_counter()
                        with mode:
                            got = cache[index]
                            cache[index] = values
                        torch.txda.current_stream().synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        gather_ok = torch.equal(got.cpu(), expected_gather)
                        allocation_ok = torch.equal(backing.cpu(), expected_backing)
                        case = {
                            "dtype": str(dtype),
                            "blocks": blocks,
                            "selected_rows": count,
                            "shape": list(cache.shape),
                            "strides": list(cache.stride()),
                            "storage_offset": cache.storage_offset(),
                            "gather_hits": mode.gather_hits,
                            "scatter_hits": mode.scatter_hits,
                            "gather_exact": gather_ok,
                            "whole_allocation_exact": allocation_ok,
                            "milliseconds": elapsed_ms,
                            "ok": gather_ok
                            and allocation_ok
                            and mode.gather_hits == mode.scatter_hits == 1,
                        }
                        report["cases"].append(case)
                        save()
                        print(json.dumps(case), flush=True)
                        if not case["ok"]:
                            raise AssertionError(
                                "product state indexing differs from the CPU oracle"
                            )
                        del (
                            expected_backing,
                            expected_cache,
                            expected_gather,
                            values,
                            values_cpu,
                            got,
                            mode,
                        )
                    del backing, cache, initial, cpu_cache
        report["ok"] = all(case["ok"] for case in report["cases"])
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    save()
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "cases": len(report["cases"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
