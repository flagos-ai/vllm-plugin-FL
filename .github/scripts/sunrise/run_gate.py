# Copyright 2026 FlagOS Contributors
"""Standalone vLLM 0.32 Sunrise V2 Gate; no repository files are changed.

Derived from codex/sunrise-vllm-0.28.0 @ b7b1258 .github/scripts/sunrise/run_gate.py.
Each invocation retains the unchanged 26-request shared Gate, then adds one
measured >=4096-token prefill. Graph acceptance uses native request-time FULL
metrics after owned-server readiness, not a V1-only replay log marker.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, suppress
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

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
        if case.get("concurrency") != count:
            raise ValueError("Incorrect Gate concurrency: " + name)
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
        for value in re.findall(
            r"(?:GPU|PTPU) KV cache size:\s*([0-9,]+)\s*tokens", text
        )
    ]
    if not capacities or min(capacities) < max_model_len:
        raise ValueError(
            "Actual KV-cache token capacity is missing or below requested context"
        )
    return {"token_capacities": capacities, "required_context": max_model_len}


def replay_evidence(text):
    # Native CUDAGraphLogging rows: unpadded | padded | padding | mode | count.
    # In V2 a FULL execution descriptor routes directly to run_fullgraph/replay.
    # Only log bytes emitted after readiness are passed to this validator.
    rows = re.findall(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*(?:CUDAGraphMode\.)?FULL\s*\|\s*(\d+)\s*\|",
        text,
    )
    events = sum(int(count) for unpadded, _, _, count in rows if int(unpadded) > 0)
    if events <= 0:
        raise ValueError("No native request-time FULL graph count after readiness")
    return {"native_full_executions": events, "sample_rows": rows[:10]}


def measured_long_prefill(
    model, served_model, port, target_tokens, max_model_len, output, base_url=None
):
    import importlib.util

    from transformers import AutoTokenizer

    if target_tokens < 4096:
        raise ValueError("Long-prefill coverage requires at least 4096 prompt tokens")
    if target_tokens > max_model_len - 256:
        raise ValueError("Long-prefill target leaves insufficient context for output")
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    header = "The audit passphrase is violet maple 42. Remember it exactly.\n"
    records = []
    while True:
        index = len(records) + 1
        records.append(
            f"Archive record {index:04d}: shelf {index % 97} contains neutral reference material for this context retention test.\n"
        )
        prompt = (
            header
            + "".join(records)
            + "\nReturn the audit passphrase from the first line. Reply with the three words only."
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) >= target_tokens:
            break
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        (base_url or f"http://127.0.0.1:{port}/v1") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with build_opener(ProxyHandler({})).open(request, timeout=300) as response:
        document = json.load(response)
    text = document["choices"][0]["message"].get("content") or ""
    gate_file = (
        Path(os.environ["SUNRISE_GATE_REPO"])
        / "tools/adaptation-gate-cases/case_utils.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_shared_sunrise_gate", gate_file
    )
    shared = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(shared)
    checks = shared.quality_checks(text, ["violet", "maple", "42"])
    measured = document.get("usage", {}).get("prompt_tokens")
    passed = (
        type(measured) is int and measured >= target_tokens and all(checks.values())
    )
    result = {
        "schema_version": 1,
        "kind": "supplemental_long_prefill",
        "model": model,
        "target_prompt_tokens": target_tokens,
        "measured_prompt_tokens": measured,
        "latency_seconds": time.monotonic() - started,
        "input": payload,
        "response": document,
        "checks": checks,
        "passed": passed,
    }
    (output / "long-prefill.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise ValueError(
            "Measured long-prefill token count or semantic response failed"
        )
    return {
        "passed": True,
        "measured_prompt_tokens": measured,
        "target_prompt_tokens": target_tokens,
        "report": "long-prefill.json",
    }


def wait_owned_service(process, port, served_model, timeout, base_url=None, model=None):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("Owned Sunrise service exited before becoming ready")
        try:
            with opener.open(
                (base_url or f"http://127.0.0.1:{port}/v1") + "/models", timeout=5
            ) as response:
                document = json.load(response)
            matches = [
                row for row in document.get("data", []) if row.get("id") == served_model
            ]
            if matches:
                if model is not None and matches[0].get("root") != model:
                    raise RuntimeError(
                        "Served model root does not match the requested model"
                    )
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
    repo=None,
):
    gate = Path(repo) / "tools/adaptation-gate-cases"
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
        "--cudagraph-metrics",
        "--allowed-local-media-path",
        str(gate / "images"),
        "--trust-remote-code",
    ]
    if mode == "eager":
        command.append("--enforce-eager")
    else:
        command += [
            "--compilation-config",
            json.dumps(
                {
                    "mode": 0,
                    "backend": "eager",
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2, 4, 8],
                }
            ),
        ]
    return command


def reuse_endpoint(url, served_model, log):
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path.rstrip("/") not in {"", "/v1"}
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or parsed.port is None
    ):
        raise ValueError(
            "Reuse endpoint must be an explicit local HTTP(S) host:port[/v1]"
        )
    if not served_model or not log or not Path(log).is_file():
        raise ValueError(
            "Reuse requires --served-model-name and an existing --server-log"
        )
    return parsed.port, f"{parsed.scheme}://{parsed.netloc}/v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--reuse-server-url")
    parser.add_argument("--server-log")
    parser.add_argument("--served-model-name")
    parser.add_argument("--long-prefill-tokens", type=int, default=4096)
    parser.add_argument("--graph-metrics-timeout", type=int, default=60)
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
    reuse = args.reuse_server_url is not None
    if reuse:
        try:
            port, base_url = reuse_endpoint(
                args.reuse_server_url, args.served_model_name, args.server_log
            )
        except ValueError as error:
            parser.error(str(error))
        served_model = args.served_model_name
        server_log = Path(args.server_log).resolve()
    else:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        served_model = "sunrise-gate-" + uuid.uuid4().hex
        base_url = f"http://127.0.0.1:{port}/v1"
        server_log = output / "server.log"
    repo = Path(args.repo).resolve(strict=True)
    os.environ["SUNRISE_GATE_REPO"] = str(repo)
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
        BASE_URL=base_url,
        RESULTS_DIR=str(output / "requests"),
        SERVER_PID_FILE=str(output / "server.pid"),
        VLLM_LOGGING_LEVEL="DEBUG",
        GEMS_VENDOR="sunrise",
        VLLM_PLUGINS="fl",
        USE_FLAGGEMS="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        TA_CLOSE_GRAPH_MEMORY="1",
        PCCL_PROTO="Simple",
    )
    for filename in (
        "run_test.sh",
        "case_utils.py",
        "test_text.py",
        "test_image.py",
        "test_mix_text_image.py",
    ):
        if not (gate / filename).is_file():
            parser.error("Missing shared Gate file: " + str(gate / filename))
    if (
        args.long_prefill_tokens < 4096
        or args.long_prefill_tokens > args.max_model_len - 256
    ):
        parser.error("Long-prefill target must be >=4096 and leave 256 context tokens")
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
        repo=repo,
    )
    facts = {
        "model": model,
        "mode": args.mode,
        "served_model_name": served_model,
        "port": port,
        "requested_configuration": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        },
        "server_command": None if reuse else command,
        "server_log": str(server_log),
        "reused_server": reuse,
        "gate_source_sha256": {
            str(path.relative_to(gate)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(gate.rglob("*"))
            if path.is_file() and (path.suffix in {".py", ".sh", ".png"})
        },
        "performance_policy": "Record request latency and token usage; no throughput or latency SLO is prescribed by the shared Gate",
        "coverage": {"shared_requests": 26, "supplemental_long_prefill_requests": 1},
        "passed": False,
    }
    server = client = None
    try:
        with ExitStack() as stack:
            gate_log = stack.enter_context((output / "gate.log").open("w"))
            if not reuse:
                log = stack.enter_context(server_log.open("w"))
                server = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                (output / "server.pid").write_text(str(server.pid))
            wait_owned_service(
                server, port, served_model, args.startup_timeout, base_url, model
            )
            facts["server_ready_unix_seconds"] = time.time()
            ready_text = server_log.read_text(errors="replace")
            if "Using Sunrise Model Runner V2" not in ready_text:
                raise ValueError(
                    "Owned server did not confirm actual Sunrise Model Runner V2"
                )
            facts["runner_v2"] = True
            facts["kv_cache_capacity"] = kv_cache_capacity(
                server_log.read_text(errors="replace"), args.max_model_len
            )
            # Exclude initialization/warmup replays from request-time evidence.
            gate_start_offset = server_log.stat().st_size
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
            facts["long_prefill"] = measured_long_prefill(
                model,
                served_model,
                port,
                args.long_prefill_tokens,
                args.max_model_len,
                output,
                base_url,
            )
            if args.mode == "graph":
                text = server_log.read_text(errors="replace")
                evidence = (
                    "Graph capturing finished" in text
                    and "Capturing CUDA graphs" in text
                )
                facts["graph_capture_evidence"] = evidence
                if not evidence:
                    raise ValueError(
                        "Graph Gate lacks successful graph capture evidence"
                    )
                with server_log.open("rb") as stream:
                    stream.seek(gate_start_offset)
                    request_text = stream.read().decode("utf-8", errors="replace")
                metrics_deadline = time.monotonic() + args.graph_metrics_timeout
                while True:
                    try:
                        facts["graph_replay_evidence"] = replay_evidence(request_text)
                        break
                    except ValueError:
                        if (
                            server is not None and server.poll() is not None
                        ) or time.monotonic() >= metrics_deadline:
                            raise
                        time.sleep(1)
                        with server_log.open("rb") as stream:
                            stream.seek(gate_start_offset)
                            request_text = stream.read().decode(
                                "utf-8", errors="replace"
                            )
            facts["passed"] = True
    except Exception as error:
        facts["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Do not kill by process name, model name, port, or unrelated GPU PID.
        client_clean = stop_owned_session(client)
        server_clean = stop_owned_session(server)
        facts["only_owned_cleanup_complete"] = client_clean and server_clean
        facts["reused_server_left_running"] = reuse
        facts["passed"] = facts["passed"] and facts["only_owned_cleanup_complete"]
        (output / "summary.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))
    return 0 if facts["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
