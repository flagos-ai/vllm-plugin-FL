"""Bounded standard-library trace I/O. No production/GPU imports."""

import atexit
import contextvars
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

MAX_FILE = 64 * 1024 * 1024
MAX_EVENT = 256 * 1024
GATE_MEMBERS = {".gitignore", "README.md", "case_utils.py", "generate_images.py", "run_serve.sh",
                "run_serve_eager.sh", "run_serve_graph.sh", "run_test.sh",
                "test_text.py", "test_image.py", "test_mix_text_image.py"} | {
                    f"images/image_{index:02d}.png" for index in range(1, 9)}
api_context = contextvars.ContextVar("musa_gate_api", default=None)
execute_context = contextvars.ContextVar("musa_gate_execute", default=None)
graph_context = contextvars.ContextVar("musa_gate_graph", default=None)
logical_context = contextvars.ContextVar("musa_gate_logical", default=None)
scenario_context = contextvars.ContextVar("musa_gate_scenario", default=None)


class Unsupported(RuntimeError):
    """An unbound source or observation capability; never a successful event."""


def unique_id():
    return uuid.uuid4().hex


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        if isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def stat_key(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def read_original(path, limit=16 * 1024 * 1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise Unsupported(f"Noncanonical input: {path}")
    first = path.lstat()
    if not stat.S_ISREG(first.st_mode) or first.st_size > limit:
        raise Unsupported(f"Not a bounded regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if stat_key(first) != stat_key(opened):
            raise Unsupported(f"Input changed before open: {path}")
        chunks = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(65536, limit + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise Unsupported(f"Input exceeded cap: {path}")
            chunks.append(chunk)
        if stat_key(opened) != stat_key(os.fstat(descriptor)):
            raise Unsupported(f"Input changed during read: {path}")
    finally:
        os.close(descriptor)
    if stat_key(first) != stat_key(path.lstat()):
        raise Unsupported(f"Input changed after read: {path}")
    body = b"".join(chunks)
    if len(body) != first.st_size:
        raise Unsupported(f"Input EOF/size mismatch: {path}")
    return body, {"path": str(path), "bytes": len(body),
                  "sha256": hashlib.sha256(body).hexdigest()}


def process_identity():
    pid = os.getpid()
    root = Path("/proc") / str(pid)
    before = (root / "stat").read_text()
    end = before.rfind(") ")
    if end < 0 or int(before[:before.find(" (")]) != pid:
        raise Unsupported("Invalid own /proc stat")
    starttime = int(before[end + 2:].split()[19])
    namespace = os.readlink(root / "ns/pid")
    if (root / "stat").read_text() != before:
        # Other stat counters change naturally; bind only the lifetime fields.
        after = (root / "stat").read_text()
        final_end = after.rfind(") ")
        if int(after[final_end + 2:].split()[19]) != starttime:
            raise Unsupported("Own process lifetime changed")
    return pid, str(starttime), namespace


class Config:
    def __init__(self):
        raw, self.original = read_original(os.environ["MUSA_GATE_OBSERVER_CONFIG"], 1024 * 1024)
        self.data = strict_json(raw)
        for name in ("observer_source_sha256", "source_binding_id"):
            if not re.fullmatch(r"[0-9a-f]{64}", self.data.get(name, "")):
                raise Unsupported(f"Unbound {name}")
        if self.data.get("schema_version") != 1 or not self.data.get("run_id"):
            raise Unsupported("Invalid observation config")
        route_id = os.environ["MUSA_GATE_OBSERVER_ROUTE_ID"]
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", route_id):
            raise Unsupported("Route ID is not a safe trace-file component")
        rows = [row for row in self.data["routes"] if row["route_id"] == route_id]
        if len(rows) != 1:
            raise Unsupported("Unbound/ambiguous route")
        self.route = rows[0]
        self.directory = Path(self.data["trace_directory"])
        if not self.directory.is_absolute():
            raise Unsupported("Trace directory must be absolute")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.resolve() != self.directory or not self.directory.is_dir():
            raise Unsupported("Noncanonical trace directory")
        own = self.data["observer_sources"]
        if digest(own) != self.data["observer_source_sha256"]:
            raise Unsupported("Observer manifest digest mismatch")
        for name, expected in own.items():
            if name not in {"__init__.py", "trace.py", "server.py", "pytest_plugin.py", "summarize.py"}:
                raise Unsupported("Unexpected observer source member")
            _, actual = read_original(Path(__file__).parent / name)
            if (actual["bytes"], actual["sha256"]) != (expected["bytes"], expected["sha256"]):
                raise Unsupported(f"Observer source changed: {name}")
        if set(own) != {"__init__.py", "trace.py", "server.py", "pytest_plugin.py", "summarize.py"}:
            raise Unsupported("Incomplete observer source manifest")

    def verify_module(self, name):
        module = sys.modules.get(name)
        expected = self.data["sources"].get(name)
        if module is None or not expected:
            raise Unsupported(f"Unbound/unloaded module: {name}")
        _, actual = read_original(module.__file__)
        if actual != expected:
            raise Unsupported(f"Module fingerprint mismatch: {name}: {actual}")
        return actual

    def verify_role(self, role, static_names=()):
        names = self.data["required_role_sources"][role]
        if not names or len(names) != len(set(names)):
            raise Unsupported("Invalid required module list")
        result = {}
        for name in names:
            if name in static_names and name not in sys.modules:
                expected = self.data["sources"][name]
                _, actual = read_original(expected["path"])
                if actual != expected:
                    raise Unsupported(f"Unloaded original source fingerprint mismatch: {name}")
                result[name] = actual
            else:
                result[name] = self.verify_module(name)
        return result

    def verify_gate_kit(self):
        members = self.data["gate_kit_files"]
        if not isinstance(members, dict) or set(members) != GATE_MEMBERS:
            raise Unsupported("Incomplete original nineteen-file Gate kit")
        root = Path(self.data["sources"]["case_utils"]["path"]).parent
        total = 0
        for name, expected in members.items():
            if expected["path"] != str(root / name):
                raise Unsupported("Original Gate member escaped its bound root")
            _, actual = read_original(expected["path"], 1024 * 1024)
            total += actual["bytes"]
            if actual != expected or total > 4 * 1024 * 1024:
                raise Unsupported(f"Original Gate kit fingerprint/cap mismatch: {name}")
        return members


class Trace:
    def __init__(self, config, role, rank=None):
        self.config, self.role, self.rank = config, role, rank
        self.pid, self.starttime, self.namespace = process_identity()
        self.process_key = f"{role}:{self.pid}:{self.starttime}"
        self.path = config.directory / f"{config.route['route_id']}-{role}-{self.pid}-{self.starttime}.jsonl"
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.seq = self.bytes = 0
        self.errors = []
        self.closed = False
        self.lock = threading.RLock()
        atexit.register(self.close)

    def emit(self, kind, data, status="ok"):
        with self.lock:
            if self.closed:
                return
            row = {"schema_version": 1, "run_id": self.config.data["run_id"],
                   "route_id": self.config.route["route_id"], "model": self.config.route["model"],
                   "mode": self.config.route["mode"], "event_id": unique_id(),
                   "pid": self.pid, "starttime": self.starttime, "pid_namespace": self.namespace,
                   "process_key": self.process_key, "seq": self.seq + 1,
                   "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                   "kind": kind, "status": status, "data": data,
                   "observer_source_sha256": self.config.data["observer_source_sha256"],
                   "source_binding_id": self.config.data["source_binding_id"]}
            try:
                body = canonical(row) + b"\n"
                limit = MAX_FILE if kind == "telemetry_end" else MAX_FILE - 8192
                if len(body) > MAX_EVENT or self.bytes + len(body) > limit:
                    raise Unsupported("Telemetry cap exceeded")
                if os.write(self.fd, body) != len(body):
                    raise Unsupported("Partial telemetry write")
                self.seq += 1
                self.bytes += len(body)
            except Exception:
                self.errors.append(traceback.format_exc())
                # Do not replace a production result/exception with telemetry failure.
                sys.stderr.write("MUSA_GATE_OBSERVER_IO_ERROR\n" + self.errors[-1])

    def activation(self, hooks, sources, status="ok", error=None, client_scope=None):
        data = {"role": self.role, "rank": self.rank,
                  "hooks": hooks, "sources_verified": status == "ok", "sources": sources,
                  "config_original": self.config.original, "error": error}
        if client_scope is not None:
            data.update(client_scope)
        self.emit("activation", data, status)
        if status != "ok":
            self.errors.append(error or status)

    def fail(self, error):
        detail = {"type": type(error).__name__, "message": str(error),
                  "traceback": traceback.format_exc()}
        self.errors.append(detail)
        self.emit("observation_error", detail, "unsupported" if isinstance(error, Unsupported) else "error")

    def close(self):
        with self.lock:
            if not self.closed:
                self.emit("telemetry_end", {"errors": list(self.errors)}, "ok" if not self.errors else "error")
                try:
                    os.close(self.fd)
                except OSError as error:
                    self.errors.append(str(error))
                    sys.stderr.write(f"MUSA_GATE_OBSERVER_CLOSE_ERROR: {error}\n")
                self.closed = True
