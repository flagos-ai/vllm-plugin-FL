# Copyright 2026 FlagOS Contributors
"""Validate the four Sunrise V2 Gate receipts and their underlying reports."""

import argparse
import hashlib
import json
from pathlib import Path

from run_gate import CHECKS, replay_evidence, validate_results

EXPECTED_MODELS = {"Qwen3.6-27B", "Qwen3.6-35B-A3B"}
REQUIRED_CONFIGURATION = {
    "tensor_parallel_size": 2,
    "max_model_len": 8192,
    "max_num_seqs": 8,
    "kv_cache_memory_bytes": 4294967296,
}


def validate_matrix(directories):
    if len(directories) != 4:
        raise ValueError("Exactly four model/mode Gate directories are required")
    found = set()
    dataset = None
    receipts = []
    for directory in directories:
        directory = Path(directory)
        receipt = json.loads((directory / "summary.json").read_text())
        key = (Path(receipt["model"]).name, receipt["mode"])
        if key[0] not in EXPECTED_MODELS or key[1] not in {"eager", "graph"}:
            raise ValueError("Unexpected model/mode Gate receipt: " + repr(key))
        if key in found:
            raise ValueError("Duplicate model/mode Gate receipt: " + repr(key))
        found.add(key)
        if any(
            receipt.get(name) is not True
            for name in ("passed", "runner_v2", "only_owned_cleanup_complete")
        ):
            raise ValueError("Gate receipt did not pass V2/quality/owned cleanup")
        if receipt.get("requested_configuration") != REQUIRED_CONFIGURATION:
            raise ValueError("Gate must use TP2, context8192, sequences8, and KV4GiB")
        gate = validate_results(
            directory / "requests",
            receipt["model"],
            receipt["served_model_name"],
            receipt["port"],
        )
        if gate != receipt.get("gate") or receipt.get("gate_returncode") != 0:
            raise ValueError("Shared Gate summary or process return code differs")
        hashes = receipt.get("gate_source_sha256")
        if not hashes or not all(
            isinstance(value, str) and len(value) == 64 for value in hashes.values()
        ):
            raise ValueError("Shared Gate source hashes are missing")
        if dataset is not None and hashes != dataset:
            raise ValueError("Gate dataset differs between model/mode scenarios")
        dataset = hashes
        long = json.loads((directory / "long-prefill.json").read_text())
        measured, target = (
            long.get("measured_prompt_tokens"),
            long.get("target_prompt_tokens"),
        )
        if (
            long.get("passed") is not True
            or long.get("model") != receipt["model"]
            or type(target) is not int
            or target < 4096
            or type(measured) is not int
            or measured < target
            or set(long.get("checks", {})) != CHECKS
            or not all(value is True for value in long["checks"].values())
            or long.get("response", {}).get("usage", {}).get("prompt_tokens")
            != measured
            or receipt.get("long_prefill", {}).get("measured_prompt_tokens") != measured
        ):
            raise ValueError("Measured long-prefill report failed or differs")
        graph = None
        if key[1] == "graph":
            log_path = Path(receipt["server_log"])
            with log_path.open("rb") as stream:
                stream.seek(receipt["gate_start_log_offset"])
                graph = replay_evidence(stream.read().decode("utf-8", errors="replace"))
            if receipt.get("graph_capture_evidence") is not True:
                raise ValueError("Missing graph capture evidence")
            if (
                receipt.get("graph_replay_evidence", {}).get(
                    "native_full_executions", 0
                )
                <= 0
            ):
                raise ValueError("Missing recorded request-time graph evidence")
        receipts.append(
            {
                "model": key[0],
                "mode": key[1],
                "directory": str(directory.resolve()),
                "shared_requests": gate["requests"],
                "measured_long_prefill_tokens": measured,
                "graph_replay_evidence": graph,
            }
        )
    expected = {
        (model, mode) for model in EXPECTED_MODELS for mode in ("eager", "graph")
    }
    if found != expected:
        raise ValueError("The four required model/mode Gates are incomplete")
    return {
        "passed": True,
        "runner": "Sunrise Model Runner V2",
        "model_mode_scenarios": 4,
        "shared_scenarios": 20,
        "shared_requests": 104,
        "supplemental_long_prefill_requests": 4,
        "total_requests": 108,
        "shared_source_digest": hashlib.sha256(
            json.dumps(dataset, sort_keys=True).encode()
        ).hexdigest(),
        "performance_policy": "Latencies and token usage recorded; no shared Gate performance SLO",
        "receipts": receipts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_matrix(args.directories)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
