# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Run the normal vLLM CLI after the opt-in M3 profile is initialized."""

import argparse
import json
import sys
from pathlib import Path

from vllm_fl.minimax_m3_config import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--port", type=int, default=18050)
    parser.add_argument("--master-port", type=int, default=29750)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    from .bootstrap import initialize

    initialize()
    if args.probe:
        return
    from vllm.reasoning import ReasoningParserManager

    reasoning = Path(__file__).with_name("reasoning.py")
    ReasoningParserManager.import_reasoning_parser(str(reasoning))
    profile = load()
    service = profile.service
    command = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        "minimax-m3",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(profile.framework.tensor_parallel_size),
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        "2",
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--port",
        str(args.port),
        "--host",
        "0.0.0.0",
        "--block-size",
        str(service["block_size"]),
        "--max-model-len",
        str(service["max_model_len"]),
        "--max-num-seqs",
        str(service["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(service["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(service["gpu_memory_utilization"]),
        "--kv-cache-dtype",
        service["kv_dtype"],
        "--attention-config",
        json.dumps({"indexer_kv_dtype": service["index_kv_dtype"]}),
        "--no-enable-prefix-caching",
        "--disable-custom-all-reduce",
        "--no-async-scheduling",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
        "--reasoning-parser",
        "minimax_m3_scoped",
        "--default-chat-template-kwargs",
        json.dumps({"thinking_mode": "adaptive"}),
        "--reasoning-parser-plugin",
        str(reasoning),
        "--compilation-config",
        json.dumps(dict(service["compilation_config"])),
        "--scheduler-cls",
        service["scheduler"],
        "--watermark",
        str(service["watermark"]),
    ]
    if args.node_rank:
        command.append("--headless")
    print("M3_SERVICE_COMMAND " + json.dumps(command), flush=True)
    sys.argv = command
    from vllm.entrypoints.cli.main import main as cli

    cli()


if __name__ == "__main__":
    main()
