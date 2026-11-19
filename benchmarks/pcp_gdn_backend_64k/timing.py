"""Six-condition matched experiment; every generate call is 65536 tokens × 1."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tp", "pcp"), required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    assert not args.output.exists()
    from vllm import LLM, SamplingParams

    settings = dict(
        model="/data/model",
        tensor_parallel_size=4,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=42,
        max_model_len=66560,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
        kv_cache_memory_bytes=4294967296,
        gpu_memory_utilization=0.85,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
        kernel_config={"moe_backend": "triton"},
        enable_chunked_prefill=True,
        language_model_only=True,
        generation_config="vllm",
        disable_log_stats=False,
        additional_config={"gdn_prefill_backend": "flashinfer"},
        worker_extension_cls="timing_worker.TimingWorker",
    )
    ids = json.loads((ROOT.parent / "qwen36_64k_20260928/prompt_ids.json").read_text())
    assert len(ids) == 65536
    requests = [{"prompt_token_ids": ids}]
    params = SamplingParams(temperature=0, seed=42, max_tokens=1, ignore_eos=True)
    result = {
        "mode": args.mode,
        "settings": settings,
        "variant_order": args.variants,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("vllm", "torch", "transformers", "triton")
        },
        "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "cells": {},
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    llm = LLM(**settings)

    def generate():
        start = time.perf_counter()
        outputs = llm.generate(requests, params, use_tqdm=False)
        elapsed = (time.perf_counter() - start) * 1000
        assert len(outputs) == 1
        request = outputs[0]
        output, metrics = request.outputs[0], request.metrics
        assert list(request.prompt_token_ids) == ids
        assert len(output.token_ids) == 1 and not metrics.is_corrupted
        return {
            "wall_ms": elapsed,
            "tokens": list(output.token_ids),
            "text": output.text,
            "corrupted": metrics.is_corrupted,
            "prefill_ms": (metrics.first_token_ts - metrics.scheduled_ts) * 1000,
            "ttft_ms": metrics.first_token_latency * 1000,
        }

    try:
        for variant in args.variants:
            norm, backend = variant.split("_", 1)
            identity = llm.collective_rpc("configure", args=(norm, backend))
            assert len(identity) == 4
            for rank in identity:
                assert rank["fa_versions"] == [3] * 10
                assert rank["norm"]["module_count"] == 101
                assert not rank["norm"]["auditing"]
                assert len(rank["gdn"]) == 30
                expected_module = (
                    "vllm_fl.cp.prstyle"
                    if args.mode == "pcp"
                    else "vllm.model_executor.models.qwen3_next"
                )
                assert rank["layer_forward"] == expected_module
                suffix = (
                    ".forward_cuda" if backend == "flashinfer" else ".forward_native"
                )
                assert all(g["dispatch"].endswith(suffix) for g in rank["gdn"])
            llm.collective_rpc("begin_core_audit")
            audit_output = generate()
            calls = llm.collective_rpc("core_audit")
            assert all(len(rank) == 30 for rank in calls)
            assert all(
                c["tokens"] == 65536 and c["sequences"] == 1 and c["chunk_metadata"]
                for rank in calls
                for c in rank
            )
            # Restore native bound methods, eliminating diagnostic wrappers.
            identity = llm.collective_rpc("configure", args=(norm, backend))
            cell = {
                "identity": identity,
                "core_audit": calls,
                "audit_output": audit_output,
                "warmups": [],
                "samples": [],
                "inspection_before": llm.collective_rpc("inspect_qwen_parallelism"),
            }
            result["cells"][variant] = cell
            save()
            for phase, count in (("warmups", args.warmups), ("samples", args.samples)):
                for index in range(count):
                    row = generate()
                    cell[phase].append(row)
                    save()
                    print(
                        args.mode,
                        variant,
                        phase,
                        index,
                        round(row["wall_ms"], 3),
                        flush=True,
                    )
            cell["inspection_after"] = llm.collective_rpc("inspect_qwen_parallelism")
            for before, after in zip(
                cell["inspection_before"], cell["inspection_after"], strict=True
            ):
                assert after["tp_size"] == 4 and len(after["gdn_layers"]) == 30
                assert all(
                    g["tp_size"] == 4 and g["out_proj_reduce_results"]
                    for g in after["gdn_layers"]
                )
                if args.mode == "pcp":
                    assert len(after["pr_layers"]) == 10
                    assert all(
                        a["prefill_calls"] - b["prefill_calls"]
                        == args.samples + args.warmups
                        for b, a in zip(
                            before["pr_layers"], after["pr_layers"], strict=True
                        )
                    )
                else:
                    assert not after["pr_layers"]
            cell["median_wall_ms"] = statistics.median(
                r["wall_ms"] for r in cell["samples"]
            )
            save()
        result["completed"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
