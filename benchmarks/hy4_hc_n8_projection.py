#!/usr/bin/env python3
"""Correctness, eager/graph replay, and skinny-N benchmark for HY4 HC.

The HY4 kernel is enabled by default; use
``VLLM_HY4_HC_N8_PROJECTION=0`` to benchmark the reference fallback.  The
script compares the complete literal expression, not projection-only timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm_fl.ops.hy4_hc_projection import _is_candidate, hc_n8_projection

K = 24576
N = 8
EPS = 1e-6
SHAPES = (1, 2, 8, 32, 64, 65, 257)


def literal(flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = flat.float()
    return F.linear(flat, weight) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + EPS
    )


def _eager_us(fn, flat: torch.Tensor, weight: torch.Tensor, repeats: int) -> float:
    for _ in range(40):
        fn(flat, weight)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            fn(flat, weight)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    return statistics.median(samples)


def _graph_us(
    fn,
    flat: torch.Tensor,
    weight: torch.Tensor,
    repeats: int,
) -> tuple[float, int, float, float]:
    # Warm up before capture so no Triton compilation happens in the graph.
    for _ in range(40):
        fn(flat, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(flat, weight)
    address = output.data_ptr()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    changed = torch.randn_like(flat)
    flat.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    replay_expected = literal(changed, weight)
    replay_diff = float((output - replay_expected).abs().max())
    return statistics.median(samples), address, float(output.abs().max()), replay_diff


def run(repeats: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260824)
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": "float32",
        "shape_k": K,
        "shape_n": N,
        "eps": EPS,
        "cases": [],
    }
    cases: list[dict[str, object]] = []
    for m in SHAPES:
        flat = torch.randn((m, K), device="cuda", dtype=torch.float32)
        weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
        expected = literal(flat, weight)
        actual = hc_n8_projection(flat, weight, EPS)
        torch.cuda.synchronize()
        diff = (actual - expected).abs()
        max_abs = float(diff.max())
        max_rel = float((diff / (expected.abs() + 1e-6)).max())
        ref_eager = _eager_us(literal, flat, weight, repeats)
        cand_eager = _eager_us(
            lambda x, w: hc_n8_projection(x, w, EPS), flat, weight, repeats
        )
        ref_graph, ref_address, _, ref_replay_diff = _graph_us(
            literal, flat, weight, repeats
        )
        cand_graph, cand_address, _, cand_replay_diff = _graph_us(
            lambda x, w: hc_n8_projection(x, w, EPS), flat, weight, repeats
        )
        cases.append(
            {
                "m": m,
                "dispatch": bool(_is_candidate(flat, weight)),
                "max_abs": max_abs,
                "max_rel": max_rel,
                "allclose_atol_rtol_3e-4": bool(
                    torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)
                ),
                "ref_eager_us": ref_eager,
                "candidate_eager_us": cand_eager,
                "eager_speedup": ref_eager / cand_eager,
                "ref_graph_us": ref_graph,
                "candidate_graph_us": cand_graph,
                "graph_speedup": ref_graph / cand_graph,
                "graph_output_address": cand_address,
                "ref_graph_output_address": ref_address,
                "graph_replay_max_abs": cand_replay_diff,
                "ref_graph_replay_max_abs": ref_replay_diff,
            }
        )
    result["cases"] = cases
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.repeats)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
