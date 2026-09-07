#!/usr/bin/env python3
"""Benchmark PLE cache row I/O on CUDA.

The benchmark runs correctness first, then 25 eager warmup iterations and 100
timed iterations.  The dedicated path uses preallocated output/mask buffers
and is also measured through CUDA graph replay.  Results are emitted as JSON
so TP8/H100 runs can be compared without parsing console output.

Example (a modest smoke case)::

    python benchmarks/benchmark_ple_state_io.py --cache-rows-list 4096,8192

For the reported 2.58-GiB cache, pass the production cache row/hidden/width
geometry explicitly.  The script intentionally allocates a transposed view,
matching the PLE path that triggered the FlagGems contiguous-copy OOM.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_fl.models.qwen3_8_flash_next.gpu.ops.ple_state import (
    ple_state_gather,
    ple_state_scatter_,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-rows-list", default="4096,8192")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=65)
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("ple_state_benchmark.json"))
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _inputs(
    cache_rows: int,
    hidden: int,
    width: int,
    rows: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The physical storage order mirrors vLLM's [blocks, width, hidden]
    # allocation; PLE consumes the non-contiguous [blocks, hidden, width] view.
    storage = torch.randn(cache_rows, width, hidden, dtype=dtype, device="cuda")
    state = storage.transpose(1, 2)
    indices = torch.arange(rows, device="cuda", dtype=torch.int64) + 1
    indices.remainder_(max(cache_rows - 1, 1))
    indices = indices.clamp_min(1)
    if rows >= 2:
        indices[1] = NULL_BLOCK_ID
    if rows >= 3:
        indices[2] = indices[0]
    # These are fixed-address buffers for the graph path.
    output = torch.empty((rows, width, hidden), dtype=dtype, device="cuda").transpose(
        1, 2
    )
    write_mask = torch.ones(rows, dtype=torch.bool, device="cuda")
    if rows >= 3:
        # Last duplicate wins; the null row is also remapped to slot zero.
        write_mask[0] = False
    return state, indices, output, write_mask


def _dedicated_step(
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    safe = torch.where(
        indices == NULL_BLOCK_ID,
        torch.zeros_like(indices),
        indices,
    )
    # Match production PLE: metadata supplies bounded/remapped indices, gather
    # returns ATen's compact result directly, and Triton writes it back.
    gathered = ple_state_gather(state, safe, indices_are_safe=True)
    ple_state_scatter_(
        state,
        safe,
        gathered,
        write_mask=write_mask,
        indices_are_safe=True,
    )


def _aten_step(state: torch.Tensor, indices: torch.Tensor) -> None:
    safe = torch.where(
        indices == NULL_BLOCK_ID,
        torch.zeros_like(indices),
        indices,
    )
    rows = torch.ops.aten.index_select.default(state, 0, safe)
    torch.ops.aten.index_copy_.default(state, 0, safe, rows)


def _python_step(state: torch.Tensor, indices: torch.Tensor) -> None:
    # This is the entry point FlagGems replaces.  Keep it as a separate named
    # baseline so the JSON records whether the serving environment is patched.
    safe = torch.where(
        indices == NULL_BLOCK_ID,
        torch.zeros_like(indices),
        indices,
    )
    rows = torch.index_select(state, 0, safe)
    state.index_copy_(0, safe, rows)


def _time_eager(step: Callable[[], None], warmup: int, iters: int) -> tuple[float, int]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed, torch.cuda.max_memory_allocated()


def _time_graph(step: Callable[[], None], warmup: int, iters: int) -> tuple[float, int]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        step()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed, torch.cuda.max_memory_allocated()


def _record(
    name: str,
    elapsed: float,
    peak: int,
    baseline_allocated: int,
    row_bytes: int,
    cache_bytes: int,
    *,
    graph: bool,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "kernel": name,
        "graph": graph,
        "status": "pass" if error is None else "error",
        "error": error,
        "latency_us": elapsed * 1e6 if error is None else None,
        "gbps": (row_bytes * 2 / elapsed / 1e9) if error is None else None,
        "row_bytes_read_write": row_bytes * 2,
        "cache_bytes": cache_bytes,
        "peak_extra_bytes": max(0, peak - baseline_allocated),
        "peak_extra_mib": max(0, peak - baseline_allocated) / 2**20,
    }


def _run_case(args: argparse.Namespace, cache_rows: int) -> dict[str, Any]:
    dtype = _dtype(args.dtype)
    state, indices, output, write_mask = _inputs(
        cache_rows, args.hidden, args.width, args.rows, dtype
    )
    row_bytes = args.rows * args.hidden * args.width * dtype.itemsize
    cache_bytes = cache_rows * args.hidden * args.width * dtype.itemsize

    # Stage 1 correctness uses independent cloned states and the direct ATen
    # reference, before any timing is attempted.
    expected_state = state.clone()
    expected_rows = torch.ops.aten.index_select.default(
        expected_state,
        0,
        torch.where(
            indices == NULL_BLOCK_ID,
            torch.zeros_like(indices),
            indices,
        ),
    )
    actual_state = state.clone()
    actual_output = torch.empty_like(output)
    ple_state_gather(actual_state, indices, output=actual_output)
    torch.testing.assert_close(actual_output, expected_rows, rtol=0, atol=0)
    ple_state_scatter_(actual_state, indices, actual_output, write_mask=write_mask)
    # The baseline's duplicate semantics are not used for validation; the
    # dedicated operation defines last-occurrence-wins explicitly.
    del expected_state, expected_rows, actual_state, actual_output

    baseline_allocated = torch.cuda.memory_allocated()
    records: list[dict[str, Any]] = []
    dedicated = lambda: _dedicated_step(state, indices, output, write_mask)
    try:
        eager_elapsed, eager_peak = _time_eager(dedicated, args.warmup, args.iters)
        records.append(
            _record(
                "ple_state_native_gather_triton_scatter",
                eager_elapsed,
                eager_peak,
                baseline_allocated,
                row_bytes,
                cache_bytes,
                graph=False,
            )
        )
        graph_elapsed, graph_peak = _time_graph(dedicated, args.warmup, args.iters)
        records.append(
            _record(
                "ple_state_native_gather_triton_scatter",
                graph_elapsed,
                graph_peak,
                baseline_allocated,
                row_bytes,
                cache_bytes,
                graph=True,
            )
        )
    except Exception as exc:
        records.append(
            _record(
                "ple_state_native_gather_triton_scatter",
                0,
                torch.cuda.max_memory_allocated(),
                baseline_allocated,
                row_bytes,
                cache_bytes,
                graph=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    for name, step in (
        ("aten_index_select_index_copy", lambda: _aten_step(state, indices)),
        ("python_index_select_index_copy", lambda: _python_step(state, indices)),
    ):
        try:
            elapsed, peak = _time_eager(step, args.warmup, args.iters)
            records.append(
                _record(
                    name,
                    elapsed,
                    peak,
                    baseline_allocated,
                    row_bytes,
                    cache_bytes,
                    graph=False,
                )
            )
        except Exception as exc:
            records.append(
                _record(
                    name,
                    0,
                    torch.cuda.max_memory_allocated(),
                    baseline_allocated,
                    row_bytes,
                    cache_bytes,
                    graph=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {
        "cache_rows": cache_rows,
        "hidden": args.hidden,
        "width": args.width,
        "rows": args.rows,
        "dtype": args.dtype,
        "row_bytes": row_bytes,
        "cache_bytes": cache_bytes,
        "results": records,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 25 or args.iters < 100:
        raise SystemExit("use at least 25 warmup and 100 timed iterations")

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "protocol": {"warmup": args.warmup, "iterations": args.iters},
        "results": [
            _run_case(args, int(value))
            for value in args.cache_rows_list.split(",")
            if value.strip()
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
