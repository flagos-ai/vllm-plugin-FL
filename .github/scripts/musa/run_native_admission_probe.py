#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Check the installed MUSA admission wrapper in one bounded native child.

This supplements model CI. Two graphs per round exercise RNG and native profiler
registry snapshot lifetime, not all 1900 slots, model throughput or profiler traces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
import signal
import stat
import sys
import sysconfig
import traceback
from pathlib import Path
from types import SimpleNamespace

IMAGE = "harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06"
PACKAGE_ROOT = "vllm_fl/dispatch/backends/vendor/musa/compilation"
SOURCES = {
    "graph": PACKAGE_ROOT + "/graph.py",
    "budget": PACKAGE_ROOT + "/musa_graph_admission.py",
}
HELPERS = ("run_native_admission_probe.py", "native_probe_utils.py", "run_ci_stage.py")
RETIRED_MODULES = (
    "vllm_fl.dispatch.backends.vendor.musa.compilation.musa_graph_exec_cache",
    "vllm_fl.dispatch.backends.vendor.musa.compilation.musa_graph_exec",
    "vllm_fl_musa_graph_exec",
)
CLASS_NAME = PACKAGE_ROOT.replace("/", ".") + ".graph.MUSAGraphWrapper"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_fingerprint(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ValueError("Source must be a canonical regular file")
    if not 0 < info.st_size <= 1024 * 1024:
        raise ValueError("Source exceeds the bounded probe scope")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise ValueError("Source changed during read")
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256_LF": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
    }


def distribution_identity(site):
    dist = metadata.distribution("vllm-plugin-fl")
    directory = Path(dist._path).resolve(strict=True)
    if directory.parent != site or not directory.name.endswith(".dist-info"):
        raise ValueError("Plugin distribution metadata is outside the private site")
    return {
        "version": dist.version,
        "metadata_directory": str(directory),
        "files": {
            name: source_fingerprint(directory / name)
            for name in ("METADATA", "RECORD")
        },
    }


def candidate_binding(candidate_root):
    root = Path(candidate_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Candidate root must be the CI checkout")
    prefix = Path(sys.prefix).resolve(strict=True)
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    if prefix == Path(sys.base_prefix).resolve(strict=True) or not site.is_relative_to(
        prefix
    ):
        raise ValueError("The probe requires the isolated installed CI interpreter")
    if site.is_relative_to(root) or root.is_relative_to(site):
        raise ValueError("Installed packages and candidate checkout must be separate")
    helper_root = Path(__file__).resolve(strict=True).parent
    if helper_root != root / ".github/scripts/musa":
        raise ValueError("Execute the probe helper from the candidate CI checkout")
    return {
        "candidate_root": str(root),
        "private_prefix": str(prefix),
        "private_site": str(site),
        "sources": {
            key: source_fingerprint(root / relative)
            for key, relative in SOURCES.items()
        },
        "plugin_distribution": distribution_identity(site),
        "helpers": {name: source_fingerprint(helper_root / name) for name in HELPERS},
        "github_identity": {
            key: os.environ.get(key)
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        },
    }


def installed_identity(binding, graph, budget):
    site = Path(binding["private_site"])
    rows = {}
    for label, module in (("graph", graph), ("budget", budget)):
        path = Path(inspect.getsourcefile(module)).resolve(strict=True)
        expected_path = site / SOURCES[label]
        if path != expected_path or path.is_relative_to(
            Path(binding["candidate_root"])
        ):
            raise ValueError(
                "Target module was not imported from the isolated installed wheel"
            )
        actual = source_fingerprint(path)
        if actual["sha256_LF"] != binding["sources"][label]["sha256_LF"]:
            raise ValueError(
                "Installed source differs from the candidate checkout: " + label
            )
        rows[label] = actual
    distribution = distribution_identity(site)
    if distribution != binding["plugin_distribution"]:
        raise ValueError("Installed plugin metadata changed during the native child")
    return rows, distribution


def worker(args, binding, contract):
    report = {
        "status": "STARTED",
        "pid": os.getpid(),
        "phases": [],
        "candidate_binding": binding,
        "github_identity": binding["github_identity"],
        "declared_image": os.environ.get("MUSA_CI_SUITE_IMAGE"),
        "scope": "Actual installed capture/replay/clear hooks with lightweight config, 2 native graphs per round",
        "full_model_setup_executed": False,
        "full_1900_cycle_certified": False,
        "configured_51_buckets_changed": False,
        "native_profiler_trace_or_update_executed": False,
        "native_profiler_registry_snapshot_exercised": False,
        "allocation_lifetime_certified": False,
    }

    def fatal(exc):
        report["status"] = "FAIL"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            contract.sealed_json(Path(args.output_dir) / "worker.json", report)
        finally:
            os._exit(1)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(
            signum,
            lambda signum, frame: fatal(
                RuntimeError(f"Owned child received signal {signum}")
            ),
        )
    sys.unraisablehook = lambda event: fatal(
        RuntimeError("Owned destructor error: " + str(event.exc_value))
    )
    guard = contract.FatalNativeContext
    try:
        import gc
        import warnings

        import torch
        import torch_musa

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.config import CUDAGraphMode

        from vllm_fl.compilation import graph as common
        from vllm_fl.dispatch.backends.vendor.musa.compilation import (
            graph as installed,
            musa_graph_admission as admission,
        )

        assert os.environ.get("MUSA_CI_SUITE_IMAGE") == IMAGE
        assert common.current_platform.device_type == "musa"
        assert torch.musa.device_count() == 4
        torch.musa.set_device(0)
        report["installed_sources"], report["installed_distribution"] = (
            installed_identity(binding, installed, admission)
        )
        cls = installed.MUSAGraphWrapper
        assert cls.__bases__ == (common.GraphWrapper,)
        assert cls.__name__ == "MUSAGraphWrapper"
        assert cls._graph_type is torch.musa.MUSAGraph
        qualname = common.current_platform.get_static_graph_wrapper_cls()
        selected_module, selected_name = qualname.rsplit(".", 1)
        assert (
            getattr(
                __import__(selected_module, fromlist=[selected_name]), selected_name
            )
            is cls
        )
        budget = admission._musa_admission_budget
        assert budget.snapshot() == dict(
            cap=1900, live=0, pending=0, total_admitted=0, epoch=0, poisoned=False
        )
        assert not any(name in sys.modules for name in RETIRED_MODULES)
        report["actual_class"] = cls.__module__ + "." + cls.__qualname__
        report["factory_qualname"] = qualname
        report["runtime"] = {
            "torch": str(torch.__version__),
            "torch_musa": str(torch_musa.__version__),
            "sys_prefix": sys.prefix,
            "visible_devices": 4,
            "device": 0,
        }
        report["context_instrumentation"] = (
            "Original native graph/device contexts are wrapped only to intercept enter/body/exit faults before context restoration"
        )
        native_device = torch.musa.device
        device_api = common.current_platform.torch_device_fn
        native_graph_context = device_api.graph
        device_api.graph = lambda *a, **kw: guard(native_graph_context(*a, **kw), fatal)

        class GuardedMusa:
            def device(self, *a, **kw):
                return guard(native_device(*a, **kw), fatal)

            def __getattr__(self, name):
                return getattr(torch.musa, name)

        class GuardedTorch:
            musa = GuardedMusa()

            def __getattr__(self, name):
                return getattr(torch, name)

        installed.torch = GuardedTorch()
        stream = torch.musa.Stream()
        kept_inputs = []
        with guard(torch.musa.stream(stream), fatal), warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapper = cls(
                lambda x: x + torch.rand_like(x),
                SimpleNamespace(compilation_config=None),
                CUDAGraphMode.PIECEWISE,
                common.GraphOptions(debug_log_enable=False, weak_ref_output=True),
            )
            set_cudagraph_capturing_enabled(True)

            def capture_round(label):
                for index in range(2):
                    value = torch.full((64,), index, dtype=torch.float32, device="musa")
                    kept_inputs.append(value)
                    # Warm the exact operation before native capture.
                    warm = value + torch.rand_like(value)
                    torch.musa.synchronize()
                    del warm
                    entry = wrapper.entry_type(batch_descriptor=f"{label}-{index}")
                    wrapper.concrete_graph_entries[index] = entry
                    wrapper._capture_graph(entry, (value,), {})
                assert budget.snapshot()["live"] == 2
                assert (
                    budget.snapshot()["pending"] == 0
                    and not budget.snapshot()["poisoned"]
                )

            def rng_sequence():
                torch.musa.manual_seed(12345)
                values = []
                for _cycle in range(2):
                    row = []
                    for entry in wrapper.concrete_graph_entries.values():
                        result = wrapper._replay_graph(
                            entry, tuple(entry.musa_input_leaves), {}
                        )
                        torch.musa.synchronize()
                        actual = result.detach().cpu().clone()
                        assert torch.isfinite(actual).all().item()
                        row.append(actual)
                    values.append(row)
                return values

            def verify_rng(label):
                first, repeated = rng_sequence(), rng_sequence()
                for cycle in range(2):
                    for index in range(2):
                        assert torch.equal(
                            first[cycle][index], repeated[cycle][index]
                        ), "Seed reset did not reproduce native replay"
                for index in range(2):
                    assert not torch.equal(first[0][index], first[1][index]), (
                        "Native replay RNG did not advance"
                    )
                report["phases"].append(
                    {
                        "phase": label,
                        "snapshot": budget.snapshot(),
                        "seed_reproduction_exact_checks": 4,
                        "RNG_advancement_checks": 2,
                    }
                )

            capture_round("first")
            verify_rng("first_capture_rng")
            owned_ids = {
                id(entry.graph) for entry in wrapper.concrete_graph_entries.values()
            }
            snapshot = torch.musa.MUSAGraph.get_latest_instance()
            assert owned_ids <= {id(graph) for graph in snapshot}
            report["native_profiler_registry_snapshot_exercised"] = True
            wrapper.clear_graphs()
            gc.collect()
            assert budget.snapshot()["live"] == 2, (
                "Externally held native graphs were incorrectly refunded"
            )
            report["phases"].append(
                {
                    "phase": "clear_with_snapshot_retained",
                    "snapshot": budget.snapshot(),
                    "external_snapshot_total_graphs": len(snapshot),
                }
            )
            del snapshot
            gc.collect()
            assert budget.snapshot()["live"] == 0, (
                "Native graph owner survived snapshot release"
            )
            assert not owned_ids.intersection(
                id(graph) for graph in torch.musa.MUSAGraph.get_latest_instance()
            )
            wrapper.clear_graphs()  # Empty clear reports the observed zero transition.
            report["phases"].append(
                {"phase": "snapshot_released", "snapshot": budget.snapshot()}
            )
            capture_round("second")
            verify_rng("second_capture_rng")
            assert budget.snapshot()["total_admitted"] == 4
            assert budget.snapshot()["cap"] == 1900
            assert not any(name in sys.modules for name in RETIRED_MODULES)
            report["retired_modules_loaded"] = []
            report["legacy_loader_loaded"] = False
            report["native_extension_loaded"] = False
            report["status"] = "PASS"
            contract.sealed_json(Path(args.output_dir) / "worker.json", report)
            # Keep second-round graphs and inputs alive; process exit is ownership boundary.
            assert kept_inputs and wrapper.concrete_graph_entries
            os._exit(0)
    except BaseException as exc:
        fatal(exc)


def report_passed(detail, binding):
    try:
        phases = detail["phases"]
        expected = [
            "first_capture_rng",
            "clear_with_snapshot_retained",
            "snapshot_released",
            "second_capture_rng",
        ]
        site = Path(binding["private_site"])
        sources = detail["installed_sources"]
        distribution = detail["installed_distribution"]
        directory = Path(distribution["metadata_directory"])
        return (
            detail["status"] == "PASS"
            and detail["candidate_binding"] == binding
            and detail["github_identity"] == binding["github_identity"]
            and detail["declared_image"] == IMAGE
            and detail["actual_class"] == CLASS_NAME
            and detail["factory_qualname"] == CLASS_NAME
            and type(detail["runtime"]["visible_devices"]) is int
            and detail["runtime"]["visible_devices"] == 4
            and type(detail["runtime"]["device"]) is int
            and detail["runtime"]["device"] == 0
            and detail["runtime"]["sys_prefix"] == binding["private_prefix"]
            and set(sources) == set(SOURCES)
            and all(
                sources[name]["sha256_LF"] == binding["sources"][name]["sha256_LF"]
                and Path(sources[name]["path"]) == site / relative
                and type(sources[name]["bytes"]) is int
                and sources[name]["bytes"] > 0
                for name, relative in SOURCES.items()
            )
            and distribution == binding["plugin_distribution"]
            and directory.parent == site
            and directory.name.endswith(".dist-info")
            and type(distribution["version"]) is str
            and bool(distribution["version"])
            and set(distribution["files"]) == {"METADATA", "RECORD"}
            and all(
                Path(distribution["files"][name]["path"]) == directory / name
                for name in ("METADATA", "RECORD")
            )
            and detail["retired_modules_loaded"] == []
            and detail["legacy_loader_loaded"] is False
            and detail["native_extension_loaded"] is False
            and detail["native_profiler_registry_snapshot_exercised"] is True
            and detail["native_profiler_trace_or_update_executed"] is False
            and detail["full_model_setup_executed"] is False
            and detail["full_1900_cycle_certified"] is False
            and detail["configured_51_buckets_changed"] is False
            and [row["phase"] for row in phases] == expected
            and all(
                type(row["snapshot"]["live"]) is int
                and row["snapshot"]["live"] == count
                and type(row["snapshot"]["cap"]) is int
                and row["snapshot"]["cap"] == 1900
                and type(row["snapshot"]["pending"]) is int
                and row["snapshot"]["pending"] == 0
                and row["snapshot"]["poisoned"] is False
                and type(row["snapshot"]["epoch"]) is int
                and row["snapshot"]["epoch"] == epoch
                and type(row["snapshot"]["total_admitted"]) is int
                and row["snapshot"]["total_admitted"] == admitted
                for row, count, epoch, admitted in zip(
                    phases, (2, 2, 0, 2), (0, 0, 1, 1), (2, 2, 2, 4)
                )
            )
            and type(phases[-1]["snapshot"]["total_admitted"]) is int
            and phases[-1]["snapshot"]["total_admitted"] == 4
            and all(
                type(phases[i]["seed_reproduction_exact_checks"]) is int
                and phases[i]["seed_reproduction_exact_checks"] == 4
                and type(phases[i]["RNG_advancement_checks"]) is int
                and phases[i]["RNG_advancement_checks"] == 2
                for i in (0, 3)
            )
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.candidate_root = str(Path(args.candidate_root).resolve(strict=True))
    args.output_dir = str(Path(args.output_dir).resolve())
    helper_root = Path(__file__).resolve().parent
    contract = load(
        "_musa_native_probe_contract", helper_root / "native_probe_utils.py"
    )
    if args.worker:
        worker(args, candidate_binding(args.candidate_root), contract)
        raise RuntimeError("Owned worker unexpectedly returned")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as output:
            output.write("gpu_stages_safe=false\n")
    try:
        if os.environ.get("MUSA_CI_SUITE_IMAGE") != IMAGE:
            raise ValueError("Fixed CI image mismatch")
        binding = candidate_binding(args.candidate_root)
    except Exception as exc:
        contract.sealed_json(
            root / "execution.json",
            {
                "status": "FAIL",
                "reason": str(exc),
                "GPU_children_started": 0,
                "later_GPU_work_permitted": False,
            },
        )
        return 1
    owned = load("_musa_native_probe_owned", helper_root / "run_ci_stage.py")
    owned.prepare_owned_reaping()
    execution = owned.bounded_run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--candidate-root",
            args.candidate_root,
            "--output-dir",
            args.output_dir,
            "--worker",
        ],
        300,
    )
    detail = None
    try:
        path = root / "worker.json"
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Native report exceeds its bounded scope")
        detail = json.loads(path.read_text(), object_pairs_hook=owned.unique_keys)
        if candidate_binding(args.candidate_root) != binding:
            raise ValueError("Candidate source changed during the probe")
    except (OSError, ValueError, AssertionError):
        detail = None
    passed = contract.execution_safe(execution) and report_passed(detail, binding)
    contract.sealed_json(
        root / "execution.json",
        {
            "status": "PASS" if passed else "FAIL",
            "execution": execution,
            "candidate_binding": binding,
            "worker": detail,
            "later_GPU_work_permitted": passed,
            "owned_reaping_is_not_GPU_idle_certificate": True,
        },
    )
    if output_file and passed:
        with open(output_file, "a") as output:
            output.write("gpu_stages_safe=true\n")
    return 0 if passed else 1


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without Python -O; native safety guards are required")
    raise SystemExit(main())
