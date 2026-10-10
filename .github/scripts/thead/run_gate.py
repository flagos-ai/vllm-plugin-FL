# Copyright 2026 FlagOS Contributors
"""Run the 26-request adaptation Gate with the versioned quality oracle."""

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# -I removes the script directory from sys.path. Load this exact sibling
# helper without changing business-package search paths.
_common_spec = importlib.util.spec_from_file_location(
    "_thead_ci_common", Path(__file__).resolve().with_name("common.py")
)
_common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)
run_command = _common.run_command
save_json = _common.save_json
stop_process_group = _common.stop_process_group
bind_process = _common.bind_process
validate_junit = _common.validate_junit

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "tools/adaptation-gate-cases"
_quality_spec = importlib.util.spec_from_file_location(
    "_thead_gate_quality", CASES / "gate_quality.py"
)
_quality = importlib.util.module_from_spec(_quality_spec)
_quality_spec.loader.exec_module(_quality)
QUALITY_ORACLE_VERSION = _quality.ORACLE_VERSION

SCENARIOS = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}
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
SERVER_LOG_CAP = 32 * 1024**2


def validate_documents(directory):
    documents = [json.loads(p.read_text()) for p in Path(directory).rglob("*.json")]
    by_scenario = {doc["case"]["scenario"]: doc for doc in documents}
    if len(documents) != 5 or set(by_scenario) != set(SCENARIOS):
        raise RuntimeError("Gate did not produce the five unique original scenarios")
    passed = requests = checks = passed_checks = 0
    for scenario, expected in SCENARIOS.items():
        doc = by_scenario[scenario]
        if doc.get("quality_oracle_version") != QUALITY_ORACLE_VERSION:
            raise RuntimeError(
                "Missing or mismatched quality oracle version: " + scenario
            )
        if len(doc["input"]) != expected or len(doc["output"]) != expected:
            raise RuntimeError("Logical request count changed: " + scenario)
        for row in doc["output"]:
            values = row.get("checks", {})
            if set(values) != CHECKS or any(
                type(value) is not bool for value in values.values()
            ):
                raise RuntimeError("Missing/invalid quality check fields: " + scenario)
            requests += 1
            checks += len(values)
            passed_checks += sum(values.values())
            passed += row.get("passed") is True and all(values.values())
    return {
        "logical_requests": requests,
        "checks": checks,
        "passed_checks": passed_checks,
        "passed_requests": passed,
        "passed": requests == 26 and checks == 260 and passed == 26,
        "quality_oracle_version": QUALITY_ORACLE_VERSION,
        "llm_explanation_profile": _quality.LLM_EXPLANATION_PROFILE,
        "other_tasks_use_exact_required_terms": True,
    }


def parse_graph_observations(stdout, stderr):
    completed = []
    stats = []
    in_stats = False
    for line in (stdout + "\n" + stderr).splitlines():
        match = re.search(r"Capturing CUDA graphs.*100%.*?(\d+)/(\d+)", line)
        if match and int(match[1]) == int(match[2]) and int(match[1]) > 0:
            completed.append(line)
        if "CUDAGraph Stats:" in line:
            in_stats = True
        row = re.search(
            r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(FULL|PIECEWISE)\s*\|\s*(\d+)\s*\|",
            line,
        )
        if in_stats and row and int(row[5]) > 0:
            stats.append({"mode": row[4], "count": int(row[5]), "raw": line})
    return {
        "completed_graph_capture_lines": list(dict.fromkeys(completed)),
        "graph_runtime_dispatch_stats": stats,
        "graph_capture_observed": bool(completed),
        "graph_runtime_dispatch_observed": bool(stats),
        "per_request_kernel_replay_count_verified": False,
    }


def owned_listener(process, port):
    """Bind /v1/models readiness to this child's actual Linux process session."""
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == f"0100007F:{port:04X}" and fields[3] == "0A":
            inodes.add(fields[9])
    if len(inodes) != 1:
        raise RuntimeError("Expected a unique localhost listening socket")
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().split(") ", 1)[1].split()
            if int(fields[2]) != process.pid or int(fields[3]) != process.pid:
                continue
            for fd in (directory / "fd").iterdir():
                if os.readlink(fd) == "socket:[" + next(iter(inodes)) + "]":
                    return {
                        "pid": int(directory.name),
                        "port": port,
                        "socket_inode": next(iter(inodes)),
                    }
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    raise RuntimeError("Ready endpoint belongs to another process")


def check_server(process, logs):
    if process.poll() is not None:
        raise RuntimeError(
            f"Own model server exited before tests: {process.returncode}"
        )
    if any(path.stat().st_size > SERVER_LOG_CAP for path in logs):
        raise RuntimeError("Own server log cap exceeded; raw logs retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.tensor_parallel_size != 4:
        parser.error("This validated Gate configuration uses TP4")
    if not (args.model / "config.json").is_file():
        parser.error("A complete local model mount is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Gate start: model={args.model} mode={args.mode}", flush=True)
    summary = {
        "model": str(args.model),
        "mode": args.mode,
        "server_process_started": False,
        "Gate_pass": False,
        "quality_oracle_version": QUALITY_ORACLE_VERSION,
        "errors": [],
        "upstream_declared_torch_2_13_requirement_satisfied": False,
    }
    # Reserve a currently unused endpoint, then verify actual listener ownership.
    # A bind race fails closed; it never sends requests to an unrelated server.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ)
    env.pop("FL_BACKEND", None)
    env.update(
        MODEL_PATH=str(args.model.resolve()),
        SERVED_MODEL_NAME="thead-ci-model",
        PORT=str(port),
        BASE_URL=f"http://127.0.0.1:{port}/v1",
        API_KEY="EMPTY",
        RESULTS_DIR=str((args.output_dir / "results").resolve()),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        VLLM_LOGGING_LEVEL="INFO",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        env["SERVED_MODEL_NAME"],
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.85",
        "--safetensors-load-strategy",
        "prefetch",
        "--trust-remote-code",
        "--allowed-local-media-path",
        str(CASES / "images"),
    ]
    argv += ["--enforce-eager"] if args.mode == "eager" else ["--cudagraph-metrics"]
    summary["server_argv"] = argv
    process = None
    logs = [
        args.output_dir / ("server." + name + ".log") for name in ("stdout", "stderr")
    ]
    handles = [path.open("xb") for path in logs]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handles[0],
            stderr=handles[1],
            start_new_session=True,
            env=env,
            cwd=REPO,
        )
        summary.update(server_process_started=True, server_pid=process.pid)
        summary["server_identity"] = bind_process(process)
        deadline = time.monotonic() + args.startup_timeout
        ready = False
        while time.monotonic() < deadline:
            check_server(process, logs)
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3
                ) as response:
                    doc = json.load(response)
                if any(row["id"] == env["SERVED_MODEL_NAME"] for row in doc["data"]):
                    summary["ready_owned_listener"] = owned_listener(process, port)
                    ready = True
                    break
            except (OSError, ValueError, KeyError):
                time.sleep(1)
        if not ready:
            raise RuntimeError("Own model server readiness timeout")
        print(f"Gate server ready: model={args.model} mode={args.mode}", flush=True)
        receipts = []
        summary["pytest_receipts"] = receipts
        # Keep all three actual pytest files, even when earlier quality checks fail.
        for name, expected in (
            ("test_text.py", 2),
            ("test_image.py", 2),
            ("test_mix_text_image.py", 1),
        ):
            check_server(process, logs)
            print(f"Gate cases start: {name}", flush=True)
            junit = (args.output_dir / (name + ".xml")).resolve()
            receipt = run_command(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-m",
                    "pytest",
                    str(CASES / name),
                    "-o",
                    "addopts=",
                    "-q",
                    "--junitxml=" + str(junit),
                ],
                args.output_dir / name.removesuffix(".py"),
                timeout=900,
                env=env,
                cwd=CASES,
                monitor=lambda: check_server(process, logs),
            )
            receipts.append(receipt)
            if not receipt["cleanup_complete"]:
                raise RuntimeError("Stop Gate after an incomplete pytest cleanup")
            try:
                validate_junit(junit, expected)
            except Exception as exc:
                summary["errors"].append(repr(exc))
        check_server(process, logs)
        summary["post_tests_owned_listener"] = owned_listener(process, port)
        summary["pytest_receipts"] = receipts
        summary["quality"] = validate_documents(args.output_dir / "results")
    except Exception as exc:
        summary["errors"].append(repr(exc))
    finally:
        if process is not None:
            try:
                stop_process_group(process)
                summary["only_owned_cleanup_complete"] = all(
                    row["cleanup_complete"]
                    for row in summary.get("pytest_receipts", [])
                )
            except Exception as exc:
                summary["errors"].append(repr(exc))
            summary["server_exit"] = process.poll()
        for handle in handles:
            handle.close()
        lines = []
        for path in logs:
            if path.stat().st_size > SERVER_LOG_CAP:
                summary["errors"].append("Server log cap exceeded: " + path.name)
            with path.open(encoding="utf8", errors="replace") as stream:
                for line in stream:
                    if re.search(r"captur|cudagraph|cuda graph", line, re.I):
                        lines.append(line.rstrip()[:2000])
        summary["graph_observation_lines"] = lines[-200:]
        # Configuration alone is not graph evidence. Retain actual capture/stat lines.
        evidence = parse_graph_observations(
            logs[0].read_text(encoding="utf8", errors="replace"),
            logs[1].read_text(encoding="utf8", errors="replace"),
        )
        summary.update(evidence)
        summary["Gate_pass"] = bool(
            summary.get("quality", {}).get("passed")
            and not summary["errors"]
            and summary.get("only_owned_cleanup_complete")
            and all(row["clean"] for row in summary.get("pytest_receipts", []))
            and (
                args.mode != "graph"
                or (
                    summary["graph_capture_observed"]
                    and summary["graph_runtime_dispatch_observed"]
                )
            )
        )
        save_json(args.output_dir / "summary.json", summary)
        quality = summary.get("quality", {})
        print(
            "Gate complete: "
            f"model={args.model} mode={args.mode} pass={summary['Gate_pass']} "
            f"requests={quality.get('passed_requests', 0)}/{quality.get('logical_requests', 0)} "
            f"checks={quality.get('passed_checks', 0)}/{quality.get('checks', 0)} "
            f"cleanup={summary.get('only_owned_cleanup_complete', False)} "
            f"capture={summary['graph_capture_observed']} "
            f"dispatch={summary['graph_runtime_dispatch_observed']}",
            flush=True,
        )
    return 0 if summary["Gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
