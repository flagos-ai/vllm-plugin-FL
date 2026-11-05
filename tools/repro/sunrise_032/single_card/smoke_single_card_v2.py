#!/usr/bin/env python3
"""Prepared offline: Qwen3-4B TP1 Sunrise V2 eager/real-graph semantic smoke.

Derived from the previously executed smoke_v2_graph.py. This new file has not
been GPU-executed during preparation. Requires explicit TANG_VISIBLE_DEVICES=4.
--help uses only stdlib. Output directory must not exist. No reset/retry/package
or container lifecycle operations are performed. A host timeout does not prove
device cleanup. An external owned watchdog must bound the actual model run.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

CHECKS = (
    (
        ("What is the capital of France? Answer with only the city name.", "paris"),
        ("Compute 1 + 1. Answer with only the number.", "2"),
    ),
    (("What is the capital of China? Answer with only the city name.", "beijing"),),
    (
        ("Compute 7 + 5. Answer with only the number.", "12"),
        ("What is the capital of France? Answer with only the city name.", "paris"),
    ),
)


class ReplayCounter:
    def __init__(self, graph):
        self.graph = graph
        self.count = 0

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def replay(self):
        self.graph.replay()
        self.count += 1


class SingleCardProof:
    def single_card_proof_start(self):
        cfg = self.vllm_config
        assert cfg.use_v2_model_runner is True
        assert type(self.model_runner).__name__ == "SunriseV2ModelRunner"
        assert cfg.parallel_config.tensor_parallel_size == 1
        mode = "eager" if cfg.model_config.enforce_eager else "graph"
        manager = self.model_runner.cudagraph_manager
        graphs = {} if manager is None else manager.graphs
        if mode == "eager":
            assert cfg.compilation_config.cudagraph_mode.name == "NONE"
            assert not graphs, "Eager mode unexpectedly captured graphs"
        else:
            assert cfg.scheduler_config.async_scheduling is True
            assert cfg.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
            assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2]
            assert manager is not None and manager._graphs_captured
            assert graphs, "No real FULL graphs captured"
            assert all("PTPUGraph" in type(graph).__name__ for graph in graphs.values())
            # Install only AFTER model readiness; capture warmups are not counted.
            manager.graphs = {
                desc: ReplayCounter(graph) for desc, graph in graphs.items()
            }
        return self.single_card_proof_snapshot()

    def single_card_proof_snapshot(self):
        cfg = self.vllm_config
        manager = self.model_runner.cudagraph_manager
        graphs = {} if manager is None else manager.graphs
        return {
            "rank": self.rank,
            "runner": type(self.model_runner).__name__,
            "runner_v2": cfg.use_v2_model_runner,
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "enforce_eager": cfg.model_config.enforce_eager,
            "async_scheduling": cfg.scheduler_config.async_scheduling,
            "graph_mode": cfg.compilation_config.cudagraph_mode.name,
            "capture_sizes": cfg.compilation_config.cudagraph_capture_sizes,
            "graphs": [
                {
                    "descriptor": str(desc),
                    "native_type": type(graph.graph).__name__,
                    "replays": graph.count,
                }
                for desc, graph in graphs.items()
            ],
        }


def validate_proof(mode, before, after):
    assert len(before) == len(after) == 1
    assert before[0]["runner"] == after[0]["runner"] == "SunriseV2ModelRunner"
    assert after[0]["runner_v2"] is True
    assert after[0]["tensor_parallel_size"] == 1
    assert after[0]["enforce_eager"] is (mode == "eager")
    if mode == "eager":
        assert before[0]["graph_mode"] == after[0]["graph_mode"] == "NONE"
        assert not before[0]["graphs"] and not after[0]["graphs"]
    else:
        assert after[0]["async_scheduling"] is True
        assert after[0]["graph_mode"] == "FULL_DECODE_ONLY"
        assert after[0]["capture_sizes"] == [1, 2]
        assert before[0]["graphs"] and after[0]["graphs"]
        assert all(graph["replays"] == 0 for graph in before[0]["graphs"])
        assert all(
            "PTPUGraph" in graph["native_type"] and graph["replays"] > 0
            for graph in after[0]["graphs"]
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("eager", "graph"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="/data/models/Qwen/Qwen3-4B", type=Path)
    args = parser.parse_args(argv)
    if os.environ.get("TANG_VISIBLE_DEVICES") != "4":
        parser.error(
            "Require explicit TANG_VISIBLE_DEVICES=4; visibility is never changed"
        )
    if args.output_dir.exists():
        parser.error(
            "output-dir must not already exist; historical receipts are preserved"
        )
    if not (args.model / "config.json").is_file():
        parser.error(
            "local model config.json must exist; this script never downloads models"
        )
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
        parser.error("this smoke requires Runner V2")
    plugins = os.environ.get("VLLM_PLUGINS", "fl")
    if "fl" not in {name.strip() for name in plugins.split(",")}:
        parser.error("VLLM_PLUGINS must include fl")
    if os.environ.get("GEMS_VENDOR", "sunrise") != "sunrise":
        parser.error("GEMS_VENDOR must be sunrise")
    if os.environ.get("TA_CLOSE_GRAPH_MEMORY", "1") != "1":
        parser.error("this variant requires TA_CLOSE_GRAPH_MEMORY=1")
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_PLUGINS=plugins,
        GEMS_VENDOR="sunrise",
        TA_CLOSE_GRAPH_MEMORY="1",
    )
    # Ensure spawned worker extension imports this new module, never the old file.
    module_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = (
        module_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    facts = {
        "mode": args.mode,
        "model": str(args.model),
        "physical_device": 4,
        "tensor_parallel_size": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": False,
        "derived_from": "smoke_v2_graph.py",
        "native_async_default_preserved": True,
    }
    receipt = args.output_dir / "receipt.json"

    def save():
        receipt.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    save()
    started = time.monotonic()
    try:
        from vllm import LLM, SamplingParams

        compile_config = {
            "mode": 0,
            "backend": "eager",
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
        }
        if args.mode == "graph":
            compile_config.update(
                cudagraph_mode="FULL_DECODE_ONLY",
                cudagraph_capture_sizes=[1, 2],
                max_cudagraph_capture_size=2,
            )
        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=1,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            enforce_eager=args.mode == "eager",
            disable_custom_all_reduce=True,
            kv_cache_memory_bytes=1073741824,
            compilation_config=compile_config,
            worker_extension_cls="smoke_single_card_v2.SingleCardProof",
        )
        assert llm.llm_engine.vllm_config.use_v2_model_runner is True
        facts["before"] = llm.collective_rpc("single_card_proof_start")
        assert len(facts["before"]) == 1
        print("V2_PROOF_BEFORE", json.dumps(facts["before"]), flush=True)
        save()
        tokenizer = llm.get_tokenizer()
        facts["requests"] = []
        for batch in CHECKS:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for question, _ in batch
            ]
            outputs = llm.generate(
                prompts, SamplingParams(temperature=0, max_tokens=16)
            )
            assert len(outputs) == len(batch)
            for (question, expected), output in zip(batch, outputs):
                answer = output.outputs[0].text.strip()
                row = {"question": question, "expected": expected, "answer": answer}
                facts["requests"].append(row)
                print(json.dumps(row), flush=True)
                save()
                assert expected in answer.casefold(), (expected, answer)
        facts["after"] = llm.collective_rpc("single_card_proof_snapshot")
        validate_proof(args.mode, facts["before"], facts["after"])
        assert len(facts["requests"]) == 5
        facts["passed"] = True
        facts["seconds"] = time.monotonic() - started
        save()
        print("SUNRISE_SINGLE_CARD_V2_PASSED", json.dumps(facts), flush=True)
    except Exception as error:
        facts.update(error=repr(error), seconds=time.monotonic() - started)
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
