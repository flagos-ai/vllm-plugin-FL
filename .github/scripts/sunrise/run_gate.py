# Copyright 2026 FlagOS Contributors
"""Run the actual shared adaptation Gate with an owned Sunrise server."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

REQUEST_IDS = {
    "text_single": ("text-llm-details",),
    "text_concurrent_8": tuple(f"text-{i:02d}" for i in range(1, 9)),
    "image_single": ("image-05",),
    "image_concurrent_8": tuple(f"image-{i:02d}" for i in range(1, 9)),
    "mixed_concurrent_8": tuple(f"text-{i:02d}" for i in range(1, 5))
    + tuple(f"image-{i:02d}" for i in range(1, 5)),
}
SCENARIOS = {name: len(ids) for name, ids in REQUEST_IDS.items()}
CHECKS = {
    "non_empty",
    "minimum_length",
    "expected_semantics",
    "expected_order",
    "no_bang_triplet",
    "no_mojibake",
    "no_control_characters",
    "no_long_character_run",
    "no_repeated_word_run",
    "no_repeated_phrase",
}


def validate_results(root, model, served_model, port):
    documents = list(Path(root).rglob("*.json"))
    found = {}
    for path in documents:
        row = json.loads(path.read_text())
        case = row["case"]
        name = case["scenario"]
        if name not in SCENARIOS or name in found:
            raise ValueError("Unexpected or duplicate Gate scenario: " + name)
        if (case["model_path"], case["served_model_name"], str(case["port"])) != (
            model,
            served_model,
            str(port),
        ):
            raise ValueError("Gate result belongs to another model/server")
        count = SCENARIOS[name]
        inputs, outputs, summary = row["input"], row["output"], row["summary"]
        if len(inputs) != count or len(outputs) != count:
            raise ValueError("Incomplete Gate request set: " + name)
        if summary != {
            "request_count": count,
            "passed_count": count,
            "failed_count": 0,
            "passed": True,
        }:
            raise ValueError("Gate quality failure: " + name)
        expected_ids = set(REQUEST_IDS[name])
        input_ids = [item["request_id"] for item in inputs]
        output_ids = [item["request_id"] for item in outputs]
        if (
            len(set(input_ids)) != count
            or len(set(output_ids)) != count
            or set(input_ids) != expected_ids
            or set(output_ids) != expected_ids
        ):
            raise ValueError(
                "Gate request/response IDs are duplicated or not the shared expected IDs: "
                + name
            )
        for output in outputs:
            checks = output.get("checks", {})
            if (
                output.get("passed") is not True
                or set(checks) != CHECKS
                or not all(value is True for value in checks.values())
            ):
                raise ValueError("Gate request quality checks did not pass: " + name)
            if not output.get("text", "").strip() or output.get("error"):
                raise ValueError("Gate response is empty or failed: " + name)
        found[name] = count
    if set(found) != set(SCENARIOS):
        raise ValueError(
            "Missing Gate scenarios: " + repr(sorted(set(SCENARIOS) - set(found)))
        )
    return {"scenarios": len(found), "requests": sum(found.values()), "passed": True}


def kv_cache_capacity(text, max_model_len):
    capacities = [
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
    ]
    if not capacities or min(capacities) < max_model_len:
        raise ValueError(
            "Actual KV-cache token capacity is missing or below requested context"
        )
    return {"token_capacities": capacities, "required_context": max_model_len}


def replay_evidence(text):
    lines = [
        line
        for line in text.splitlines()
        if "Sunrise graph replay completed" in line
        and re.search(
            r"runtime_mode=(?:CUDAGraphMode\.)?FULL\b",
            line,
        )
    ]
    if not lines:
        raise ValueError(
            "No actual FULL Sunrise graph replay during shared Gate requests"
        )
    return {"events": len(lines), "sample": lines[:10]}


def wait_owned_service(process, port, served_model, timeout):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned Sunrise service exited before becoming ready")
        try:
            with opener.open(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                document = json.load(response)
            if any(row.get("id") == served_model for row in document.get("data", [])):
                return
            raise RuntimeError("Port serves a different model/server")
        except URLError:
            time.sleep(1)
    raise TimeoutError("Owned Sunrise service did not become ready")


def session_members(session):
    """Only processes still in the server/client session belong to this run."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            if int(fields[3]) == session and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned_session(process):
    if process is None:
        return True
    for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
        for pid in session_members(process.pid):
            with suppress(ProcessLookupError):
                os.kill(pid, sig)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and session_members(process.pid):
            time.sleep(0.1)
        if not session_members(process.pid):
            process.wait(timeout=5)
            return True
    return False


def server_command(
    model,
    served_model,
    port,
    mode,
    output,
    kv_bytes,
    tp_size,
    max_model_len=8192,
    max_num_seqs=8,
):
    gate = Path(__file__).resolve().parents[3] / "tools/adaptation-gate-cases"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--disable-custom-all-reduce",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        "0.92",
        "--kv-cache-memory-bytes",
        str(kv_bytes),
        "--skip-mm-profiling",
        "--allowed-local-media-path",
        str(gate / "images"),
        "--trust-remote-code",
    ]
    if mode == "eager":
        command.append("--enforce-eager")
    else:
        command += [
            "--compilation-config",
            json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}),
        ]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=4294967296)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=10800)
    args = parser.parse_args()
    model = str(Path(args.model).resolve())
    if not (Path(model) / "config.json").is_file():
        parser.error("Model must contain config.json: " + model)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    served_model = "sunrise-gate-" + uuid.uuid4().hex
    repo = Path(__file__).resolve().parents[3]
    gate = repo / "tools/adaptation-gate-cases"
    env = os.environ.copy()
    if env.get("VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS", "0") != "0":
        parser.error("Zero-output GDN benchmarking is forbidden in correctness Gates")
    env["NO_PROXY"] = env.get("NO_PROXY", "") + ",127.0.0.1,localhost"
    env["no_proxy"] = env.get("no_proxy", "") + ",127.0.0.1,localhost"
    env.update(
        MODEL_PATH=model,
        SERVED_MODEL_NAME=served_model,
        PORT=str(port),
        BASE_URL=f"http://127.0.0.1:{port}/v1",
        RESULTS_DIR=str(output / "requests"),
        SERVER_PID_FILE=str(output / "server.pid"),
        VLLM_LOGGING_LEVEL="DEBUG",
        GEMS_VENDOR="sunrise",
        VLLM_PLUGINS="fl",
        USE_FLAGGEMS="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        TA_CLOSE_GRAPH_MEMORY="1",
        PCCL_PROTO="Simple",
    )
    command = server_command(
        model,
        served_model,
        port,
        args.mode,
        output,
        args.kv_cache_memory_bytes,
        args.tensor_parallel_size,
        args.max_model_len,
        args.max_num_seqs,
    )
    facts = {
        "model": model,
        "mode": args.mode,
        "port": port,
        "server_command": command,
        "passed": False,
    }
    server = client = None
    try:
        with (
            (output / "server.log").open("w") as log,
            (output / "gate.log").open("w") as gate_log,
        ):
            server = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (output / "server.pid").write_text(str(server.pid))
            wait_owned_service(server, port, served_model, args.startup_timeout)
            facts["kv_cache_capacity"] = kv_cache_capacity(
                (output / "server.log").read_text(errors="replace"), args.max_model_len
            )
            # Exclude initialization/warmup replays from request-time evidence.
            gate_start_offset = (output / "server.log").stat().st_size
            facts["gate_start_log_offset"] = gate_start_offset
            # This is the shared Gate, including all three actual pytest files.
            client = subprocess.Popen(
                ["bash", str(gate / "run_test.sh")],
                cwd=gate,
                env=env,
                stdout=gate_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            facts["gate_returncode"] = client.wait(timeout=args.timeout)
            facts["gate"] = validate_results(
                output / "requests", model, served_model, port
            )
            if facts["gate_returncode"]:
                raise ValueError("Shared Gate process failed")
            if args.mode == "graph":
                text = (output / "server.log").read_text(errors="replace")
                evidence = (
                    "Graph capturing finished" in text
                    and "Capturing CUDA graphs" in text
                )
                facts["graph_capture_evidence"] = evidence
                if not evidence:
                    raise ValueError(
                        "Graph Gate lacks successful graph capture evidence"
                    )
                with (output / "server.log").open("rb") as stream:
                    stream.seek(gate_start_offset)
                    request_text = stream.read().decode("utf-8", errors="replace")
                facts["graph_replay_evidence"] = replay_evidence(request_text)
            facts["passed"] = True
    except Exception as error:
        facts["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Do not kill by process name, model name, port, or unrelated GPU PID.
        client_clean = stop_owned_session(client)
        server_clean = stop_owned_session(server)
        facts["only_owned_cleanup_complete"] = client_clean and server_clean
        facts["passed"] = facts["passed"] and facts["only_owned_cleanup_complete"]
        (output / "summary.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))
    return 0 if facts["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
