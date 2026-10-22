#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""One owned native MUSA synthetic comparison. Never a model throughput claim."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
import warnings
import weakref
from collections import OrderedDict
from pathlib import Path

SIZES = (1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16))
PIECES = 121
CAP = 1900
POLICIES = ("current", "admission", "public_drop", "native")
IMAGE = "harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06"
SOURCE_ROOT = "vllm_fl/dispatch/backends/vendor/musa/compilation"
EXPECTED_SOURCE_SHA256 = {
    "musa_graph_exec_cache.py": "02edc16e958da763c09a730731e42c1ce45c4945bb4139ff54415ac779b5f0f0",
    "musa_graph_exec.py": "e5a7e7c7aa537c6e3fdbae0777fc2bfef523e0e2d62d6e7567740d8a9b21d593",
    "musa_graph_exec.cpp": "a64485ebc9da93958db53b21c78bbcc043e2ef88fa3e98624cf21ced2dad97d9",
    "graph.py": "48542e99bc61c3a1d1937a68a2cfd7e3bb7b821f1350eb8c91f5bc979daa3da0",
}
COUNTS = (
    "capture",
    "native_replay",
    "eager",
    "policy_sync",
    "free_exec",
    "instantiate",
    "public_reset",
    "ops_loader",
)


def sealed_json(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # Exclusive link publishes a complete report without overwriting evidence.
    os.link(partial, path)
    partial.unlink()


def source_identity(root):
    rows = {}
    if not EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Comparison source guards were not frozen")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = root / SOURCE_ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            # Git/checkout line endings may differ; guard committed source bytes.
            normalized = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if normalized != expected:
                raise RuntimeError(f"Pinned source mismatch: {name}")
        rows[name] = {"sha256": expected, "physical_sha256": actual}
    return rows


def load_source(root, filename, name):
    spec = importlib.util.spec_from_file_location(name, root / SOURCE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def delta(before, after):
    return {key: after[key] - before[key] for key in before}


def run_variant(backend, policy, state, cache_module=None, load_ops=None):
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    assert len(SIZES) == 51 and len(SIZES) * PIECES == 6171
    counts = state["explicit_python_operations"]
    entries = []
    live = OrderedDict()
    cache = None
    if policy == "current":
        if cache_module is None or load_ops is None:
            raise ValueError("Current policy requires exact production cache")
        cache = cache_module._MusaGraphExecCache(
            backend.policy_device, load_ops, cap=CAP
        )
    state["phase"] = "capture"
    start = time.perf_counter()
    for bucket, size in enumerate(SIZES):
        for piece in range(PIECES):
            index = len(entries)
            value = index % 23
            entry = {
                "bucket": bucket,
                "size": size,
                "piece": piece,
                "input": backend.input(size, value),
                "graph": None,
                "output": None,
                "value": value,
            }
            state["capture_requested"] += 1
            if policy == "admission" and len(live) >= CAP:
                output = backend.eager(entry["input"])
                counts["eager"] += 1
                state["capture_admission_missed"] += 1
                entry["output"] = None
                entries.append(entry)
                # Correctness is a separate, explicitly synchronizing pass.
                del output
                continue
            if policy == "public_drop" and len(live) >= CAP:
                _, victim = live.popitem(last=False)
                backend.policy_sync()
                counts["policy_sync"] += 1
                old_graph = victim["graph"]
                old_reference = weakref.ref(old_graph)
                victim["output"] = None
                victim["graph"] = None
                backend.reset(old_graph)
                counts["public_reset"] += 1
                del old_graph
                if old_reference() is not None:
                    gc.collect()
                if old_reference() is not None:
                    raise RuntimeError("Dropped native graph remains strongly owned")
                state["public_drop_graphs_released"] += 1
            graph = backend.graph()
            state["native_capture_attempted"] += 1
            if cache is None:
                output = backend.capture(graph, entry["input"])
            else:
                with cache.capture(graph):
                    output = backend.capture(graph, entry["input"])
            counts["capture"] += 1
            state["native_capture_completed"] += 1
            entry["graph"] = graph
            entry["output"] = output
            entries.append(entry)
            if policy != "current":
                live[index] = entry
            # Drop loop temporaries; only the owning entry retains graph/output.
            del graph, output, entry
    backend.measure_sync()
    state["capture_batch_elapsed_seconds"] = time.perf_counter() - start
    state["capture_memory"] = backend.memory()
    state["retained_templates"] = sum(entry["graph"] is not None for entry in entries)
    resident_indices = set()
    if cache is not None:
        resident_indices = {
            index
            for index, entry in enumerate(entries)
            if id(entry["graph"]) in cache._live
        }
    else:
        resident_indices = set(live)
    state["resident_execs"] = len(resident_indices)
    if policy == "native":
        assert len(resident_indices) == 6171
    else:
        assert len(resident_indices) == CAP
    state["resident_pieces_per_bucket"] = [
        sum(
            index in resident_indices
            for index in range(bucket * PIECES, (bucket + 1) * PIECES)
        )
        for bucket in range(51)
    ]
    # Capture records work; it need not execute it. Replay residents first.
    # Cold templates and unadmitted/dropped pieces run the original eager body.
    state["phase"] = "capture_correctness"
    start = time.perf_counter()
    state["postcapture_validation_outcomes"] = {"native_replay": 0, "eager": 0}
    for entry in entries:
        graph = entry["graph"]
        native = graph is not None and backend.replay(graph, cache)
        if native:
            output = entry["output"]
            counts["native_replay"] += 1
            state["postcapture_validation_outcomes"]["native_replay"] += 1
        else:
            output = backend.eager(entry["input"])
            counts["eager"] += 1
            state["postcapture_validation_outcomes"]["eager"] += 1
        backend.check(output, entry["size"], entry["value"] * 2 + 3)
        state["capture_exact_checks"] += 1
    state["capture_correctness_seconds"] = time.perf_counter() - start
    assert state["capture_exact_checks"] == 6171
    # Two cycles retain each exact input Tensor object. Timed batch enqueues
    # and end-sync are separate from per-case exact validation below.
    for cycle in (1, 2):
        state["phase"] = f"runtime_{cycle}_timed"
        row = {"cycle": cycle, "native_replay": 0, "eager": 0, "exact_checks": 0}
        before = counts.copy()
        start = time.perf_counter()
        for index, entry in enumerate(entries):
            backend.fill(entry["input"], (index + cycle) % 23)
        backend.measure_sync()
        # Input preparation is excluded from the timed enqueue/end-sync batch.
        start = time.perf_counter()
        for index, entry in enumerate(entries):
            graph = entry["graph"]
            native = graph is not None and backend.replay(graph, cache)
            if native:
                counts["native_replay"] += 1
                row["native_replay"] += 1
            else:
                output = backend.eager(entry["input"])
                counts["eager"] += 1
                row["eager"] += 1
                del output
        backend.measure_sync()
        row["synthetic_batch_enqueue_and_end_sync_seconds"] = (
            time.perf_counter() - start
        )
        row["explicit_python_ops_delta"] = delta(before, counts)
        # Correctness pass repeats the same work, outside the timing.
        state["phase"] = f"runtime_{cycle}_correctness"
        start = time.perf_counter()
        for index, entry in enumerate(entries):
            graph = entry["graph"]
            native = graph is not None and backend.replay(graph, cache)
            if native:
                output = entry["output"]
                counts["native_replay"] += 1
            else:
                output = backend.eager(entry["input"])
                counts["eager"] += 1
            backend.check(output, entry["size"], ((index + cycle) % 23) * 2 + 3)
            row["exact_checks"] += 1
        row["correctness_seconds"] = time.perf_counter() - start
        assert row["exact_checks"] == 6171
        assert (row["native_replay"], row["eager"]) == (
            (6171, 0) if policy == "native" else (1900, 4271)
        )
        row["memory"] = backend.memory()
        state["runtime_cycles"].append(row)
    state["phase"] = "completed"
    assert state["native_capture_completed"] == (
        1900 if policy == "admission" else 6171
    )
    assert state["capture_requested"] == 6171
    assert all(row["exact_checks"] == 6171 for row in state["runtime_cycles"])
    if policy != "current":
        assert counts["ops_loader"] == 0
        assert "_comparison_production_loader" not in sys.modules
        assert "vllm_fl_musa_graph_exec" not in sys.modules
    state["loader_used"] = counts["ops_loader"] > 0
    state["status"] = "PASS"
    # Return refs to keep graph pools alive through final report publication.
    return entries, cache


class FatalNativeContext:
    def __init__(self, native, fail):
        self.native = native
        self.fail = fail

    def __enter__(self):
        try:
            return self.native.__enter__()
        except BaseException as exc:
            self.fail(exc)

    def __exit__(self, kind, value, tb):
        if kind is not None:
            self.fail(value)
        try:
            return self.native.__exit__(kind, value, tb)
        except BaseException as exc:
            self.fail(exc)


def fail_immediately(state, output, exc, exit_process=os._exit):
    state["status"] = "FAIL"
    state["failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        sealed_json(Path(output) / "result.json", state)
    finally:
        # Even CPU report IO failure cannot unwind live GPU-owning frames.
        exit_process(1)


class NativeBackend:
    def __init__(self, device, state, fail):
        import torch
        import torch_musa

        self.torch = torch
        self.api = torch.musa
        self.state = state
        self.fail = fail
        sys.unraisablehook = lambda event: self.fail(
            RuntimeError(
                "Owned native object destructor failed: " + str(event.exc_value)
            )
        )
        if self.api.device_count() != 4:
            raise RuntimeError(
                "Comparison requires exactly four visible allocated devices"
            )
        self.api.set_device(device)
        self.device = torch.device("musa", device)
        self.stream = self.api.Stream()
        self.pool = self.api.graph_pool_handle()
        self.policy_device = self.PolicyDevice(self)
        state["runtime"] = {
            "torch": str(torch.__version__),
            "torch_musa": str(torch_musa.__version__),
            "torch_musa_runtime": str(getattr(torch.version, "musa", "unavailable")),
            "torch_git": str(getattr(torch.version, "git_version", "unavailable")),
            "visible_device_count": 4,
            "device_ordinal": device,
            "device_name": str(self.api.get_device_name(device)),
            "declared_container_image": os.environ.get("MUSA_CI_SUITE_IMAGE"),
        }
        versions = {}
        for name in ("version.json", "version.txt"):
            path = Path("/usr/local/musa") / name
            if path.is_file() and path.stat().st_size <= 65536:
                data = path.read_bytes()
                versions[name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "text": data.decode("utf-8", errors="replace")[:4096],
                }
        state["runtime"]["SDK_version_files"] = versions
        state["runtime"]["driver_version"] = (
            "Refer to the original native allocation/preflight log; no extra driver command is run"
        )
        self.input(1, 1)
        self.measure_sync()
        state["initial_memory"] = self.memory()

    class PolicyDevice:
        def __init__(self, backend):
            self.backend = backend

        def current_device(self):
            return self.backend.api.current_device()

        def is_current_stream_capturing(self):
            return self.backend.api.is_current_stream_capturing()

        def device(self, ordinal):
            return FatalNativeContext(
                self.backend.api.device(ordinal), self.backend.fail
            )

        def synchronize(self):
            self.backend.state["explicit_python_operations"]["policy_sync"] += 1
            self.backend.api.synchronize()

    def input(self, size, value):
        with FatalNativeContext(self.api.stream(self.stream), self.fail):
            return self.torch.full(
                (size,), value, dtype=self.torch.float32, device=self.device
            )

    def fill(self, tensor, value):
        with FatalNativeContext(self.api.stream(self.stream), self.fail):
            tensor.fill_(value)

    def graph(self):
        return self.api.MUSAGraph()

    def capture(self, graph, tensor):
        with (
            FatalNativeContext(self.api.stream(self.stream), self.fail),
            FatalNativeContext(
                self.api.graph(graph, pool=self.pool, stream=self.stream), self.fail
            ),
        ):
            return tensor * 2 + 3

    def eager(self, tensor):
        with FatalNativeContext(self.api.stream(self.stream), self.fail):
            return tensor * 2 + 3

    def replay(self, graph, cache=None):
        with FatalNativeContext(self.api.stream(self.stream), self.fail):
            if cache is not None:
                return cache.replay(graph)
            graph.replay()
            return True

    def reset(self, graph):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                graph.reset()
            except BaseException as exc:
                self.fail(exc)

    def policy_sync(self):
        self.api.synchronize()

    def measure_sync(self):
        self.state["measurement_barriers"] += 1
        self.api.synchronize()

    def memory(self):
        return {
            "allocated": int(self.api.memory_allocated()),
            "reserved": int(self.api.memory_reserved()),
        }

    def check(self, tensor, size, expected):
        with FatalNativeContext(self.api.stream(self.stream), self.fail):
            actual = tensor.detach().cpu()
            self.state["validator_d2h_checks"] += 1
            want = self.torch.full((size,), expected, dtype=self.torch.float32)
            if not self.torch.equal(actual, want):
                raise AssertionError(
                    f"Native exact tensor result mismatch at size={size}"
                )


def initial_state(policy):
    return {
        "schema": 1,
        "policy": policy,
        "status": "STARTED",
        "phase": "initialization",
        "synthetic_only": True,
        "model_wrapper_count_certified": False,
        "model_throughput_certified": False,
        "capture_sizes": list(SIZES),
        "pieces_per_size": PIECES,
        "exec_cap": CAP,
        "capture_requested": 0,
        "native_capture_attempted": 0,
        "native_capture_completed": 0,
        "capture_admission_missed": 0,
        "capture_exact_checks": 0,
        "public_drop_graphs_released": 0,
        "runtime_cycles": [],
        "explicit_python_operations": {key: 0 for key in COUNTS},
        "measurement_barriers": 0,
        "validator_d2h_checks": 0,
        "timing_scope": "synthetic batch enqueue plus explicit end synchronization; per-case correctness is separate; not model throughput",
        "counter_scope": "explicit Python policy-maintenance only; validator, native graph internals and implicit SDK synchronization are not fully observable",
        "contract_certificates": {
            "borrowed_outputs": False,
            "input_mutation_alias": False,
            "RNG": False,
            "native_profiler": False,
            "installed_wheel_execution": False,
            "SDK_internal_op_counts": False,
        },
        "loader_entrypoint": "Exact committed source _build_ops (current only), bypassing public loader device gate; _build_ops retains its original version/header/compiler guards",
        "source_baseline_commit": "9ea227b705c3bcf77e94ce70e2502a2fc2f2e8f6",
        "public_drop_boundary": "reset/delete removes cold templates/RNG graph ownership; native releasePool may retain reserved storage; not a memory-release or model-equivalence certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-root", default=str(Path(__file__).resolve().parents[3])
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = initial_state(args.policy)
    state["pid"] = os.getpid()
    state["cpu_owned_evidence_only_after_failure"] = True

    def fail(exc):
        fail_immediately(state, output, exc)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(
            signum,
            lambda signum, frame: fail(
                RuntimeError(f"Owned worker received signal {signum}")
            ),
        )

    try:
        if os.environ.get("MUSA_CI_SUITE_IMAGE") != IMAGE:
            raise RuntimeError("Fixed comparison container image mismatch")
        root = Path(args.source_root).resolve(strict=True)
        state["source_identity"] = source_identity(root)
        state["harness_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        state["source_set_digest"] = hashlib.sha256(
            json.dumps(EXPECTED_SOURCE_SHA256, sort_keys=True).encode()
        ).hexdigest()
        state["github_identity"] = {
            key: os.environ.get(key)
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        }
        backend = NativeBackend(args.device, state, fail)
        cache_module = None
        loader = None
        if args.policy == "current":
            cache_module = load_source(
                root, "musa_graph_exec_cache.py", "_comparison_production_cache"
            )

            def loader():
                state["explicit_python_operations"]["ops_loader"] += 1
                module = load_source(
                    root, "musa_graph_exec.py", "_comparison_production_loader"
                )
                ops = module._build_ops()

                class OpsProxy:
                    def has_exec(self, graph):
                        return ops.has_exec(graph)

                    def free_exec(self, graph):
                        state["explicit_python_operations"]["free_exec"] += 1
                        return ops.free_exec(graph)

                    def inst_exec(self, graph):
                        state["explicit_python_operations"]["instantiate"] += 1
                        return ops.inst_exec(graph)

                return OpsProxy()

        kept = run_variant(backend, args.policy, state, cache_module, loader)
        assert kept and state["status"] == "PASS"
        sealed_json(output / "result.json", state)
        # Process exit is the ownership boundary; no post-report GPU cleanup.
        os._exit(0)
    except BaseException as exc:
        fail(exc)


if __name__ == "__main__":
    main()
