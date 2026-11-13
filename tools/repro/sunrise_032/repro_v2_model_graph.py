#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Full-model native Runner V2 TP2 graph symptom reproducer (not GPU-retested).

Derived from the separately archived Qwen3-4B smoke. This uses the actual V2
model/sampling/executor queues, unlike repro_tp2_graph.py; matching symptoms do
not establish a common root cause. It does not add fences or synchronous
scheduling. Capture types are checked, but request replay counts are not measured.
--help imports no accelerator packages. No retry, reset or external process
management is performed; an external timeout cannot guarantee device cleanup.
"""

import argparse
import json
import os
import uuid
from pathlib import Path

from repro_tp2_graph import emit, environment_updates, versions, visible_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen3-4B model path")
    parser.add_argument("--visible-devices", required=True, type=visible_pair)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ta-close-graph-memory", choices=("1", "unset"), default="1")
    return parser


def describe_capture(worker):
    """Read actual worker state; do not wrap or change graph replay."""
    runner = worker.model_runner
    manager = runner.cudagraph_manager
    assert worker.vllm_config.use_v2_model_runner
    assert manager is not None and manager._graphs_captured and manager.graphs
    return {
        "rank": worker.rank,
        "runner": type(runner).__name__,
        "graph_mode": str(manager.cudagraph_mode),
        "capture_count": len(manager.graphs),
        "captured_token_counts": sorted({desc.num_tokens for desc in manager.graphs}),
        "native_graph_types": [
            type(graph).__name__ for graph in manager.graphs.values()
        ],
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        updates = environment_updates(args, os.environ)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        return parser.error(str(exc))
    if (
        not Path(args.model).is_dir()
        or not (Path(args.model) / "config.json").is_file()
    ):
        parser.error("--model must be a local model directory containing config.json")
    os.environ.update(updates)  # Only this user-invoked process and its children.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"repro-v2-model-graph-{uuid.uuid4().hex}.json"
    receipt = {
        "portable_refactor": True,
        "variant": "native_v2_async_model_graph",
        "model": args.model,
        "visible_devices": os.environ["TANG_VISIBLE_DEVICES"],
        "TA_CLOSE_GRAPH_MEMORY": os.environ.get("TA_CLOSE_GRAPH_MEMORY"),
        "PCCL_PROTO": os.environ["PCCL_PROTO"],
        "versions": versions(),
        "request_replay_counted": False,
    }
    emit("provenance", **receipt, result_path=str(result_path))
    try:
        from vllm import LLM, SamplingParams
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.platforms import current_platform

        assert current_platform.device_type == "ptpu"
        llm = LLM(
            model=args.model,
            tensor_parallel_size=2,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            enforce_eager=False,
            disable_custom_all_reduce=True,
            async_scheduling=True,
            kv_cache_memory_bytes=1073741824,
            compilation_config={
                "mode": 0,
                "backend": "eager",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2],
                "max_cudagraph_capture_size": 2,
            },
        )
        cfg = llm.llm_engine.vllm_config
        assert cfg.use_v2_model_runner
        assert cfg.scheduler_config.async_scheduling is True
        assert cfg.compilation_config.mode == CompilationMode.NONE
        assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        assert sorted(cfg.compilation_config.cudagraph_capture_sizes) == [1, 2]
        captures = llm.collective_rpc(describe_capture)
        assert len(captures) == 2
        for worker in captures:
            assert worker["runner"] == "SunriseV2ModelRunner"
            assert worker["graph_mode"] == "FULL_DECODE_ONLY"
            assert all("PTPUGraph" in name for name in worker["native_graph_types"])
        receipt["captures"] = captures
        emit("model_ready", runner_v2=True, async_scheduling=True, captures=captures)
        checks = [
            [
                (
                    "What is the capital of France? Answer with only the city name.",
                    "paris",
                ),
                ("Compute 1 + 1. Answer with only the number.", "2"),
            ],
            [
                (
                    "What is the capital of China? Answer with only the city name.",
                    "beijing",
                ),
            ],
            [
                ("Compute 7 + 5. Answer with only the number.", "12"),
                (
                    "What is the capital of France? Answer with only the city name.",
                    "paris",
                ),
            ],
        ]
        tokenizer = llm.get_tokenizer()
        answers = []
        for batch_id, batch in enumerate(checks):
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for question, _ in batch
            ]
            emit("batch_begin", batch=batch_id, requests=len(batch))
            outputs = llm.generate(
                prompts, SamplingParams(temperature=0, max_tokens=16)
            )
            assert len(outputs) == len(batch)
            for (question, expected), output in zip(batch, outputs):
                answer = output.outputs[0].text.strip()
                emit("answer", batch=batch_id, expected=expected, answer=answer)
                assert expected in answer.casefold(), (expected, answer)
                answers.append({"question": question, "answer": answer})
        assert len(answers) == 5
        receipt.update(status="semantic_outputs_passed", answers=answers)
    except Exception as exc:
        receipt.update(
            status="failed",
            error=str(exc),
            automatic_retry=False,
            native_device_cleanup_guaranteed=False,
        )
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        emit(
            "failed",
            error=str(exc),
            result_path=str(result_path),
            automatic_retry=False,
            native_device_cleanup_guaranteed=False,
        )
        raise
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    emit(
        "semantic_outputs_passed",
        requests=5,
        result_path=str(result_path),
        request_replay_counted=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
