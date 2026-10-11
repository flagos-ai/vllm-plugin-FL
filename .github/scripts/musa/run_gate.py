"""Linux CI orchestration of four routes using the unchanged original tests."""

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_gate import aggregate
from gate_common import (
    ROUTES,
    SCENARIOS,
    TEST_FILES,
    fingerprint,
    read_json,
    require,
    write_json,
)

LOG_CAP = 64 * 1024 * 1024
START_SECONDS = 30
READY_SECONDS = 600
PYTEST_SECONDS = 1900
CLIENT_SECONDS = 5800
ROUTE_SECONDS = 6900
TERM_SECONDS = 15
API_SHUTDOWN_SECONDS = 60
API_TERM_SECONDS = 90
KILL_SECONDS = 15
INTERRUPTED = False


def interrupted(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True


def check_time(deadline):
    require(not INTERRUPTED, "CI runner received a termination signal")
    require(time.monotonic() < deadline, "Route or stage deadline expired")


def proc_text(path, cap=32768):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        body = os.read(fd, cap + 1)
        require(len(body) <= cap and not os.read(fd, 1), "Oversized proc metadata")
        return body.decode("utf-8")
    finally:
        os.close(fd)


def proc_stat(pid):
    raw = proc_text(f"/proc/{pid}/stat", 8192)
    fields = raw[raw.rfind(")") + 2 :].split()
    require(len(fields) >= 20, "Incomplete process stat")
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime": int(fields[19]),
    }


def identity(pid):
    first = proc_stat(pid)
    status = proc_text(f"/proc/{pid}/status")
    uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
    namespaces = {
        name: os.readlink(f"/proc/{pid}/ns/{name}") for name in ("pid", "mnt", "user")
    }
    last = proc_stat(pid)
    require(
        all(
            first[key] == last[key]
            for key in ("pid", "ppid", "pgid", "sid", "starttime")
        ),
        "Process lifetime changed during metadata sampling",
    )
    return {key: first[key] for key in ("pid", "ppid", "pgid", "sid", "starttime")} | {
        "uid": int(uid_line.split()[1]),
        "namespaces": namespaces,
        "lifetime_stable": True,
    }


def session_processes(sid):
    entries = [entry.name for entry in os.scandir("/proc") if entry.name.isdigit()]
    require(len(entries) <= 4096, "CI process inventory exceeds bounded scan")
    result = []
    for name in entries:
        try:
            preliminary = proc_stat(int(name))
            if preliminary["sid"] == sid and preliminary["state"] != "Z":
                current = identity(int(name))
                require(
                    current["starttime"] == preliminary["starttime"],
                    "Session member changed lifetime",
                )
                result.append(current)
        except (FileNotFoundError, ProcessLookupError):
            # A disappearing unrelated/transient PID never becomes a positive attestation.
            continue
    return result


def verify_known_lifetimes(known, stat_reader=proc_stat, identity_reader=identity):
    """A previously attested live child cannot disappear by changing its SID."""
    live = []
    for previous in known.values():
        try:
            preliminary = stat_reader(previous["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(
            preliminary["starttime"] == previous["starttime"],
            "Known child PID was reused; remaining ownership is UNKNOWN",
        )
        if preliminary["state"] == "Z":
            continue
        current = identity_reader(previous["pid"])
        require(
            all(
                current[key] == previous[key]
                for key in ("pid", "starttime", "pgid", "sid", "uid", "namespaces")
            ),
            "Known live child escaped its attested session/group/namespace",
        )
        live.append(current)
    return live


@contextlib.contextmanager
def startup_deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError("Subprocess startup deadline expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class OwnedProcess:
    """Retain the unreaped session leader until all owned group cleanup finishes.

    Signals require the original leader's current starttime, parent, PGID and SID.
    Unknown ownership is a cleanup failure, never permission to kill another job.
    """

    def __init__(self, argv, env, cwd, log_root, term_seconds=TERM_SECONDS):
        self.term_seconds = term_seconds
        self.argv = argv
        self.out_path = log_root / "stdout.raw"
        self.err_path = log_root / "stderr.raw"
        self.out = self.out_path.open("xb")
        self.err = self.err_path.open("xb")
        self.process = subprocess.Popen.__new__(subprocess.Popen)
        self.initial = None
        self.pidfd = None
        self.start_error = None
        self.known = {}
        self.last_group_sample = 0.0
        try:
            with startup_deadline(START_SECONDS):
                subprocess.Popen.__init__(
                    self.process,
                    argv,
                    env=env,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=self.out,
                    stderr=self.err,
                    start_new_session=True,
                )
                self.initial = identity(self.process.pid)
                require(
                    self.initial["ppid"] == os.getpid()
                    and self.initial["pid"]
                    == self.initial["pgid"]
                    == self.initial["sid"],
                    "New subprocess is not an owned session",
                )
                self.pidfd = os.pidfd_open(self.process.pid, 0)
                self.known[(self.initial["pid"], self.initial["starttime"])] = (
                    self.initial
                )
        except Exception:
            self.start_error = traceback.format_exc()
            # Popen may have forked before a startup failure. Keep the actual PID.
            pid = getattr(self.process, "pid", None)
            if type(pid) is int and self.initial is None:
                try:
                    candidate = identity(pid)
                    require(
                        candidate["ppid"] == os.getpid()
                        and candidate["pid"] == candidate["pgid"] == candidate["sid"],
                        "Failed startup did not retain an owned session",
                    )
                    self.initial = candidate
                    self.pidfd = os.pidfd_open(pid, 0)
                    self.known[(pid, candidate["starttime"])] = candidate
                except Exception:
                    self.start_error += (
                        "\nOwnership recovery failed:\n" + traceback.format_exc()
                    )

    def exit_observation(self):
        pid = getattr(self.process, "pid", None)
        if type(pid) is not int:
            return None
        # WNOWAIT keeps the original leader available for safe process-group cleanup.
        result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if result is None:
            return None
        return (
            result.si_status if result.si_code == os.CLD_EXITED else -result.si_status
        )

    def logs_within_cap(self):
        require(
            self.out_path.stat().st_size <= LOG_CAP
            and self.err_path.stat().st_size <= LOG_CAP,
            "Raw subprocess log cap exceeded",
        )

    def signal_owned(self, signum, group=False):
        require(
            self.initial is not None and self.pidfd is not None,
            "No actual owned subprocess lifetime",
        )
        current = proc_stat(self.initial["pid"])
        require(
            all(
                current[key] == self.initial[key]
                for key in ("pid", "ppid", "pgid", "sid", "starttime")
            ),
            "Original unreaped session leader lifetime changed",
        )
        if group:
            os.killpg(self.initial["pgid"], signum)
        else:
            signal.pidfd_send_signal(self.pidfd, signum, None, 0)

    def members(self):
        if self.initial is None:
            return []
        live = verify_known_lifetimes(self.known)
        current = session_processes(self.initial["sid"])
        for row in current:
            self.known.setdefault((row["pid"], row["starttime"]), row)
        combined = {(row["pid"], row["starttime"]): row for row in [*live, *current]}
        return list(combined.values())

    def sample_owned_children(self):
        if time.monotonic() - self.last_group_sample >= 1:
            self.members()
            self.last_group_sample = time.monotonic()

    def close(self):
        errors = []
        signals = []
        actual_exit = None
        remaining = []
        remaining_verified = False
        forced_kill = False
        try:
            if self.initial is None:
                require(
                    getattr(self.process, "pid", None) is None,
                    "Startup left a PID with UNKNOWN ownership",
                )
            else:
                for signum, grace in (
                    (signal.SIGTERM, self.term_seconds),
                    (signal.SIGKILL, KILL_SECONDS),
                ):
                    members = self.members()
                    live = [
                        row for row in members if proc_stat(row["pid"])["state"] != "Z"
                    ]
                    if not live:
                        break
                    if signum == signal.SIGKILL:
                        require(
                            all(row["pgid"] == self.initial["pgid"] for row in live),
                            "Owned descendants escaped the original process group",
                        )
                    # TERM only the actual owned API/entrypoint leader. vLLM then
                    # performs EngineCore and worker.shutdown cleanup itself.
                    group = signum == signal.SIGKILL
                    self.signal_owned(signum, group=group)
                    forced_kill = forced_kill or group
                    signals.append({"signal": signum, "monotonic": time.monotonic()})
                    deadline = time.monotonic() + grace
                    while time.monotonic() < deadline:
                        live = [
                            row
                            for row in self.members()
                            if proc_stat(row["pid"])["state"] != "Z"
                        ]
                        if not live:
                            break
                        time.sleep(0.2)
                remaining = [
                    row
                    for row in self.members()
                    if proc_stat(row["pid"])["state"] != "Z"
                ]
                remaining_verified = True
                require(not remaining, "Owned subprocess descendants remain alive")
                actual_exit = self.process.wait(timeout=1)
                if forced_kill:
                    errors.append(
                        f"Normal owned-process shutdown exceeded TERM{self.term_seconds}; KILL15 fallback was required"
                    )
        except Exception:
            errors.append(traceback.format_exc())
        finally:
            if self.pidfd is not None:
                os.close(self.pidfd)
            self.out.close()
            self.err.close()
        log_refs = {}
        for name, path in (("stdout", self.out_path), ("stderr", self.err_path)):
            try:
                log_refs[name] = fingerprint(path, LOG_CAP)
            except Exception:
                errors.append(traceback.format_exc())
                log_refs[name] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": None,
                    "complete_fingerprint": False,
                }
        return {
            "closed": not errors,
            "actual_exit": actual_exit,
            "actual_exit_known": type(actual_exit) is int,
            "signals": signals,
            "forced_kill": forced_kill,
            "remaining_processes": remaining,
            "remaining_observation_complete": remaining_verified
            if self.initial is not None
            else not errors,
            "errors": errors,
            "stdout": log_refs["stdout"],
            "stderr": log_refs["stderr"],
        }


def run_command(argv, env, cwd, directory, seconds, route_deadline, sample=None):
    directory.mkdir()
    report = {
        "argv": argv,
        "attempted": True,
        "actual_exit": None,
        "timed_out": False,
        "errors": [],
    }
    owned = OwnedProcess(argv, env, cwd, directory)
    report["identity"] = owned.initial
    try:
        require(owned.start_error is None, owned.start_error or "")
        deadline = min(route_deadline, time.monotonic() + seconds)
        while owned.exit_observation() is None:
            check_time(deadline)
            owned.logs_within_cap()
            owned.sample_owned_children()
            if sample:
                sample()
            time.sleep(0.2)
    except Exception as error:
        report["timed_out"] = (
            isinstance(error, TimeoutError) or "deadline" in str(error).lower()
        )
        report["errors"].append(traceback.format_exc())
    finally:
        report["cleanup"] = owned.close()
        report["actual_exit"] = report["cleanup"]["actual_exit"]
        report["stdout"] = report["cleanup"]["stdout"]
        report["stderr"] = report["cleanup"]["stderr"]
        report["errors"].extend(report["cleanup"]["errors"])
    write_json(directory / "command.json", report)
    return report


def cleanup_complete(cleanup):
    return (
        cleanup.get("closed") is True
        and cleanup.get("errors") == []
        and cleanup.get("remaining_observation_complete") is True
        and cleanup.get("remaining_processes") == []
    )


def route_cleanup_complete(route):
    if route["server"].get("attempted") and not cleanup_complete(route["cleanup"]):
        return False
    preflight = route.get("preflight_command")
    if preflight is None or not cleanup_complete(preflight.get("cleanup", {})):
        return False
    return all(
        not item.get("attempted") or cleanup_complete(item.get("cleanup", {}))
        for item in route["pytest"]
    )


def route_env(args, route, route_root):
    env = os.environ.copy()
    cache = Path(args.cache_root).resolve(strict=True)
    for name in ("triton", "flaggems", "torchinductor"):
        (cache / name).mkdir(exist_ok=True)
    env.update(
        {
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "FLAGGEMS_CACHE_DIR": str(cache / "flaggems"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "MODEL_PATH": route["model_path"],
            "PORT": str(route["port"]),
            "SERVED_MODEL_NAME": route["served_model_name"],
            "RESULTS_DIR": str(route_root / "results"),
            "BASE_URL": f"http://127.0.0.1:{route['port']}/v1",
            "REQUEST_TIMEOUT": "300",
            "MAX_TOKENS": "128",
        }
    )
    # Explicit task configuration; inherited loader/vendor environment is preserved.
    env["VLLM_PLUGINS"] = "fl,musa_gate_observer" if args.observer_config else "fl"
    if args.observer_config:
        env.update(
            {
                "MUSA_GATE_OBSERVER_CONFIG": args.observer_config,
                "MUSA_GATE_OBSERVER_ROUTE_ID": route["route_id"],
            }
        )
    return env


def one_route(args, route, root, binding_id):
    route_root = root / route["route_id"]
    route_root.mkdir()
    logs = route_root / "logs"
    logs.mkdir()
    env = route_env(args, route, route_root)
    route.update(
        {
            "source_binding_id": binding_id,
            "errors": [],
            "pytest": [],
            "scenarios": {},
            "ready": {"ready": False, "attempts": []},
            "server": {"attempted": False, "started": False, "identity": None},
            "cleanup": {"closed": True, "errors": [], "actual_exit": None},
            "environment": {
                key: env[key]
                for key in (
                    "MODEL_PATH",
                    "PORT",
                    "SERVED_MODEL_NAME",
                    "RESULTS_DIR",
                    "BASE_URL",
                    "REQUEST_TIMEOUT",
                    "MAX_TOKENS",
                    "VLLM_PLUGINS",
                    "TRITON_CACHE_DIR",
                    "FLAGGEMS_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            },
        }
    )
    deadline = time.monotonic() + ROUTE_SECONDS
    preflight_output = route_root / "preflight.json"
    preflight_argv = [
        args.python,
        "-I",
        "-B",
        str(Path(__file__).with_name("preflight.py")),
        "--repo",
        args.repo,
        "--expected-commit",
        args.expected_commit,
        "--gate-dir",
        args.gate_dir,
        "--cache-root",
        args.cache_root,
        "--source-binding",
        args.source_binding,
        "--output",
        str(preflight_output),
    ]
    server = None
    samples = {}
    sample_errors = []
    last_sample = 0.0

    def sample():
        nonlocal last_sample
        if (
            server is None
            or server.initial is None
            or time.monotonic() - last_sample < 1
        ):
            return
        server.logs_within_cap()
        last_sample = time.monotonic()
        try:
            for row in server.members():
                require(
                    row["starttime"] >= server.initial["starttime"]
                    and row["uid"] == server.initial["uid"],
                    "Server session member has an unrelated lifetime or UID",
                )
                key = (row["pid"], row["starttime"])
                if key not in samples:
                    require(
                        len(samples) < 4096, "Owned process attestation cap reached"
                    )
                    samples[key] = row | {
                        "first_seen_monotonic": last_sample,
                        "last_seen_monotonic": last_sample,
                    }
                else:
                    require(
                        samples[key]["namespaces"] == row["namespaces"]
                        and samples[key]["pgid"] == row["pgid"],
                        "Server process namespace/group changed",
                    )
                    samples[key]["last_seen_monotonic"] = last_sample
        except Exception:
            sample_errors.append(traceback.format_exc())

    try:
        preflight_env = env.copy()
        # Telemetry is enabled only for the actual server and original pytest clients.
        preflight_env["VLLM_PLUGINS"] = "fl"
        preflight_env.pop("MUSA_GATE_OBSERVER_CONFIG", None)
        preflight_env.pop("MUSA_GATE_OBSERVER_ROUTE_ID", None)
        preflight = run_command(
            preflight_argv,
            preflight_env,
            args.repo,
            route_root / "preflight-command",
            300,
            deadline,
        )
        route["preflight_command"] = preflight
        route["preflight"] = fingerprint(preflight_output)
        require(
            preflight["actual_exit"] == 0 and not preflight["errors"],
            "Actual normal-site source/cache/device preflight failed",
        )
        require(
            read_json(preflight_output)["passed"] is True,
            "Device preflight did not pass",
        )
        require(Path(route["model_path"]).is_dir(), "Model directory is absent")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", route["port"]))
        # Preserve the venv executable path: resolving its symlink would select base Python.
        executable = str(Path(args.python).parent / "vllm")
        argv = [
            executable,
            "serve",
            route["model_path"],
            "--served-model-name",
            route["served_model_name"],
            "--host",
            "127.0.0.1",
            "--port",
            str(route["port"]),
            "--tensor-parallel-size",
            "4",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "4096",
            "--gpu-memory-utilization",
            ("0.92" if route["mode"] == "graph" else "0.6"),
            "--trust-remote-code",
            "--shutdown-timeout",
            str(API_SHUTDOWN_SECONDS),
            "--allowed-local-media-path",
            str(Path(args.gate_dir) / "images"),
        ]
        if route["mode"] == "eager":
            argv.append("--enforce-eager")
        else:
            # Cover mixed prefill/decode batches up to the scheduler budget.
            # The default is derived from max_num_seqs and can stop at 16.
            argv.extend(
                [
                    "--compilation-config",
                    json.dumps(
                        {
                            "cudagraph_capture_sizes": [
                                1,
                                2,
                                4,
                                8,
                                16,
                                256,
                                1024,
                                4096,
                            ],
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
        route["server"]["argv"] = argv
        route["server"]["attempted"] = True
        server_logs = logs / "server"
        server_logs.mkdir()
        server = OwnedProcess(
            argv, env, args.repo, server_logs, term_seconds=API_TERM_SECONDS
        )
        require(server.start_error is None, server.start_error or "")
        route["server"].update({"started": True, "identity": server.initial})
        (logs / "server.pid").write_text(
            str(server.initial["pid"]) + "\n", encoding="ascii"
        )
        route["server"]["pid_file"] = fingerprint(logs / "server.pid")
        sample()
        ready_deadline = min(deadline, time.monotonic() + READY_SECONDS)
        while not route["ready"]["ready"]:
            check_time(ready_deadline)
            server.logs_within_cap()
            require(server.exit_observation() is None, "Server exited before readiness")
            sample()
            observation = {"monotonic": time.monotonic(), "status": None, "error": None}
            try:
                # Disable proxy routing only for this explicit local health connection.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(
                    f"http://127.0.0.1:{route['port']}/health", timeout=3
                ) as response:
                    observation["status"] = response.status
                    route["ready"]["ready"] = response.status == 200
            except Exception as error:
                observation["error"] = f"{type(error).__name__}: {error}"
            route["ready"]["attempts"].append(observation)
            if not route["ready"]["ready"]:
                time.sleep(1)
        client_deadline = min(deadline, time.monotonic() + CLIENT_SECONDS)
        client_blocked = None
        for filename in TEST_FILES:
            junit = route_root / f"{filename}.junit.xml"
            command = [
                args.python,
                "-I",
                "-B",
                "-m",
                "pytest",
                "-sv",
                str(Path(args.gate_dir) / filename),
                f"--junitxml={junit}",
            ]
            if args.observer_config:
                command.extend(["-p", "musa_gate_observer.pytest_plugin"])
            try:
                require(client_blocked is None, client_blocked or "")
                check_time(client_deadline)
                result = run_command(
                    command,
                    env,
                    args.gate_dir,
                    route_root / filename,
                    PYTEST_SECONDS,
                    client_deadline,
                    sample,
                )
                result["file"] = filename
                if result["identity"] is not None:
                    row = result["identity"] | {
                        "owner": "pytest",
                        "entrypoint": filename,
                    }
                    samples[(row["pid"], row["starttime"])] = row
                result["junit"] = (
                    fingerprint(junit, 16 * 1024 * 1024) if junit.exists() else None
                )
                route["pytest"].append(result)
                if not cleanup_complete(result["cleanup"]):
                    client_blocked = f"NOT_RUN: preceding {filename} has UNKNOWN/incomplete process cleanup"
            except Exception:
                route["pytest"].append(
                    {
                        "file": filename,
                        "argv": command,
                        "attempted": False,
                        "actual_exit": None,
                        "timed_out": False,
                        "errors": [traceback.format_exc()],
                    }
                )
            # Each original file is still attempted after an earlier ordinary pytest failure.
        for scenario in SCENARIOS:
            result_path = (
                route_root
                / "results"
                / route["model"]
                / f"port-{route['port']}"
                / f"{scenario}.json"
            )
            if result_path.exists():
                route["scenarios"][scenario] = fingerprint(result_path)
    except Exception:
        route["errors"].append(traceback.format_exc())
    finally:
        try:
            sample()
        except Exception:
            route["errors"].append(traceback.format_exc())
        if server is not None:
            route["cleanup"] = server.close()
        route["process_attestations"] = write_json(
            route_root / "process-attestations.json",
            {
                "schema_version": 1,
                "route_id": route["route_id"],
                "server_identity": route["server"]["identity"],
                "processes": list(samples.values()),
                "errors": sample_errors,
                "scope": "Independent stable /proc sampling of the spawned server session",
            },
        )
        route["errors"].extend(sample_errors)
        for filename in TEST_FILES:
            if not any(item["file"] == filename for item in route["pytest"]):
                route["pytest"].append(
                    {
                        "file": filename,
                        "attempted": False,
                        "actual_exit": None,
                        "timed_out": False,
                        "errors": [
                            "Server/preflight/readiness did not permit this entry"
                        ],
                    }
                )
        write_json(route_root / "route.json", route)
    return route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--model-27b", default="/data/models/Qwen3.6-27B")
    parser.add_argument("--model-35b", default="/data/models/Qwen3.6-35B-A3B")
    parser.add_argument("--ports", default="18062,18063,18064,18065")
    parser.add_argument("--observer-config")
    parser.add_argument("--observer-source-sha256")
    parser.add_argument("--observer-events-dir")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    require(
        sys.platform == "linux"
        and hasattr(os, "pidfd_open")
        and hasattr(signal, "pidfd_send_signal")
        and hasattr(os, "WNOWAIT")
        and not sys.flags.no_site,
        "Run this CI draft on Linux with normal-site Python and pidfd/WNOWAIT",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_commit),
        "Commit must be a full Git OID",
    )
    args.python = str(Path(args.python).absolute())
    require(Path(args.python).is_file(), "Python executable is absent")
    for name in ("repo", "gate_dir", "source_binding", "cache_root"):
        setattr(args, name, str(Path(getattr(args, name)).resolve(strict=True)))
    ports = [int(value) for value in args.ports.split(",")]
    require(
        len(ports) == 4
        and len(set(ports)) == 4
        and all(1 <= value <= 65535 for value in ports),
        "Expected four unique legal ports",
    )
    root = Path(args.output_root).absolute()
    require(
        root.parent.resolve(strict=True) == root.parent, "Output parent is noncanonical"
    )
    root.mkdir()  # exclusive namespace; existing evidence is never reused
    observer_source = None
    if args.observer_config:
        args.observer_config = str(Path(args.observer_config).resolve(strict=True))
        config = read_json(args.observer_config)
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.observer_source_sha256 or ""),
            "Observer source hash is required",
        )
        require(
            config["observer_source_sha256"] == args.observer_source_sha256,
            "Observer config/source hash mismatch",
        )
        observer_source = args.observer_source_sha256
        trace_dir = root / "observer-traces"
        trace_dir.mkdir()
        if args.observer_events_dir is not None:
            require(
                Path(args.observer_events_dir).absolute() == trace_dir,
                "Observer trace directory must be this exclusive run's observer-traces",
            )
        args.observer_events_dir = str(trace_dir)
        require(
            Path(config["trace_directory"]).absolute() == trace_dir,
            "Producer config must point at this exclusive run's trace directory",
        )
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    binding = fingerprint(args.source_binding, 64 * 1024 * 1024)
    run_id = args.run_id or uuid.uuid4().hex
    require(
        re.fullmatch(r"[0-9a-f]{32}", run_id), "Run ID must be a 32-character UUID hex"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": args.expected_commit,
        "source_binding": binding,
        "source_binding_id": binding["sha256"],
        "observer_source_sha256": observer_source,
        "observer_config": fingerprint(args.observer_config)
        if args.observer_config
        else None,
        "observer_command": None,
        "routes": [],
        "errors": [],
        "budgets_seconds": {
            "startup": START_SECONDS,
            "readiness": READY_SECONDS,
            "pytest_each": PYTEST_SECONDS,
            "client_route": CLIENT_SECONDS,
            "route": ROUTE_SECONDS,
            "term": TERM_SECONDS,
            "kill": KILL_SECONDS,
            "server_shutdown": API_SHUTDOWN_SECONDS,
            "server_term": API_TERM_SECONDS,
        },
    }
    if args.observer_config:
        config = read_json(args.observer_config)
        require(
            config["run_id"] == manifest["run_id"],
            "Observer config must bind the actual CI --run-id",
        )
    blocked_cleanup = None
    for index, (route_id, model, mode, default_port) in enumerate(ROUTES):
        route = {
            "route_id": route_id,
            "model": model,
            "mode": mode,
            "model_path": args.model_27b if index < 2 else args.model_35b,
            "served_model_name": model,
            "port": ports[index],
        }
        if INTERRUPTED or blocked_cleanup is not None:
            reason = "runner terminated" if INTERRUPTED else blocked_cleanup
            manifest["routes"].append(
                route
                | {
                    "source_binding_id": binding["sha256"],
                    "errors": ["NOT_RUN: " + reason],
                    "pytest": [],
                    "scenarios": {},
                    "ready": {"ready": False},
                    "server": {"started": False, "identity": None},
                    "cleanup": {"closed": True, "errors": []},
                }
            )
            continue
        try:
            completed = one_route(args, route, root, binding["sha256"])
            manifest["routes"].append(completed)
            if not route_cleanup_complete(completed):
                blocked_cleanup = (
                    f"preceding route {route_id} has incomplete owned-process cleanup"
                )
                manifest["errors"].append(blocked_cleanup)
        except Exception:
            # An evidence/IO failure in one route cannot turn the remaining matrix into a pass.
            manifest["routes"].append(
                route
                | {
                    "source_binding_id": binding["sha256"],
                    "errors": [traceback.format_exc()],
                    "pytest": [],
                    "scenarios": {},
                    "ready": {"ready": False},
                    "server": {"started": False, "identity": None},
                    "cleanup": {
                        "closed": False,
                        "errors": ["Route evidence finalization failed"],
                    },
                }
            )
            blocked_cleanup = (
                f"preceding route {route_id} has UNKNOWN owned-process cleanup"
            )
            manifest["errors"].append(blocked_cleanup)
    manifest_path = root / "route-manifest.json"
    write_json(root / "route-manifest-before-observer.json", manifest)
    observer_path = None
    if args.observer_config:
        try:
            require(
                args.observer_events_dir is not None, "No observer raw events directory"
            )
            events = sorted(
                Path(args.observer_events_dir).resolve(strict=True).glob("*.jsonl")
            )
            require(0 < len(events) <= 128, "Missing or excessive observer event files")
            observer_path = root / "observer-summary.json"
            argv = [
                args.python,
                "-I",
                "-B",
                "-m",
                "musa_gate_observer.summarize",
                "--trace-dir",
                args.observer_events_dir,
                "--expected-manifest",
                args.observer_config,
                *(part for row in ROUTES for part in ("--route", row[0])),
                "--output",
                str(observer_path),
            ]
            observation = run_command(
                argv,
                os.environ.copy(),
                args.repo,
                root / "observer-command",
                300,
                time.monotonic() + 330,
            )
            manifest["observer_command"] = fingerprint(
                root / "observer-command" / "command.json"
            )
            require(
                observation["actual_exit"] == 0 and not observation["errors"],
                "Independent observer summarizer failed",
            )
        except Exception:
            write_json(
                root / "observer-failure.json", {"errors": [traceback.format_exc()]}
            )
            observer_path = None
    write_json(manifest_path, manifest)
    result = aggregate(manifest_path, observer_path)
    write_json(root / "gate-summary.json", result)
    print(f"Gate passed: {result['gate_passed']}; evidence: {root}", flush=True)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
