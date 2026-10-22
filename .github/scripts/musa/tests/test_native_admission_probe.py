#!/usr/bin/env python3
"""CPU report/failure controls for the small native CI probe, without target imports."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load("_native_admission_report_test", ROOT / "run_native_admission_probe.py")
utils = load("_native_admission_utils_test", ROOT / "native_probe_utils.py")


def complete_report():
    site = Path("D:/private-probe/venv/site-packages")
    metadata = site / "vllm_plugin_fl-0.0.0.dist-info"
    distribution = {
        "version": "0.0.0",
        "metadata_directory": str(metadata),
        "files": {
            name: {"path": str(metadata / name), "bytes": 16, "sha256_LF": "b" * 64}
            for name in ("METADATA", "RECORD")
        },
    }
    binding = {
        "candidate_root": "D:/candidate",
        "private_prefix": "D:/private-probe/venv",
        "private_site": str(site),
        "plugin_distribution": distribution,
        "sources": {
            name: {
                "path": str(Path("D:/candidate") / relative),
                "bytes": 16,
                "sha256_LF": "a" * 64,
            }
            for name, relative in probe.SOURCES.items()
        },
        "helpers": {},
        "github_identity": {
            "GITHUB_SHA": "c" * 40,
            "GITHUB_RUN_ID": "1",
            "GITHUB_RUN_ATTEMPT": "1",
        },
    }
    phases = []
    for name, live, epoch, admitted in zip(
        (
            "first_capture_rng",
            "clear_with_snapshot_retained",
            "snapshot_released",
            "second_capture_rng",
        ),
        (2, 2, 0, 2),
        (0, 0, 1, 1),
        (2, 2, 2, 4),
    ):
        phase = {
            "phase": name,
            "snapshot": {
                "live": live,
                "cap": 1900,
                "pending": 0,
                "poisoned": False,
                "epoch": epoch,
                "total_admitted": admitted,
            },
        }
        if name in ("first_capture_rng", "second_capture_rng"):
            phase.update(seed_reproduction_exact_checks=4, RNG_advancement_checks=2)
        phases.append(phase)
    detail = {
        "status": "PASS",
        "candidate_binding": copy.deepcopy(binding),
        "github_identity": copy.deepcopy(binding["github_identity"]),
        "declared_image": probe.IMAGE,
        "actual_class": probe.CLASS_NAME,
        "factory_qualname": probe.CLASS_NAME,
        "runtime": {
            "visible_devices": 4,
            "device": 0,
            "sys_prefix": binding["private_prefix"],
        },
        "installed_sources": {
            name: {"path": str(site / relative), "bytes": 16, "sha256_LF": "a" * 64}
            for name, relative in probe.SOURCES.items()
        },
        "installed_distribution": copy.deepcopy(distribution),
        "retired_modules_loaded": [],
        "legacy_loader_loaded": False,
        "native_extension_loaded": False,
        "native_profiler_registry_snapshot_exercised": True,
        "native_profiler_trace_or_update_executed": False,
        "full_model_setup_executed": False,
        "full_1900_cycle_certified": False,
        "configured_51_buckets_changed": False,
        "phases": phases,
    }
    return detail, binding


class ReportControls(unittest.TestCase):
    def test_complete_small_report_only(self):
        detail, binding = complete_report()
        self.assertTrue(probe.report_passed(detail, binding))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_musa", sys.modules)

    def test_partial_or_inaccurate_report_cannot_unlock_next_gpu_stage(self):
        def change(detail, kind):
            if kind == "missing":
                del detail["phases"]
            elif kind == "source":
                detail["installed_sources"]["graph"]["sha256_LF"] = "f" * 64
            elif kind == "checkout_import":
                detail["installed_sources"]["graph"]["path"] = "D:/candidate/graph.py"
            elif kind == "metadata":
                detail["installed_distribution"]["version"] = "unexpected"
            elif kind == "identity":
                detail["github_identity"]["GITHUB_SHA"] = "f" * 40
            elif kind == "alias":
                detail["actual_class"] = "MUSAAdmissionGraphWrapper"
            elif kind == "loader":
                detail["retired_modules_loaded"] = [probe.RETIRED_MODULES[0]]
            elif kind == "refund":
                detail["phases"][1]["snapshot"]["live"] = 0
            elif kind == "leak":
                detail["phases"][2]["snapshot"]["live"] = 2
            elif kind == "recovery":
                detail["phases"][3]["snapshot"]["total_admitted"] = 2
            elif kind == "epoch":
                detail["phases"][3]["snapshot"]["epoch"] = 0
            elif kind == "pending":
                detail["phases"][0]["snapshot"]["pending"] = 1
            elif kind == "poison":
                detail["phases"][3]["snapshot"]["poisoned"] = True
            elif kind == "cap":
                detail["phases"][0]["snapshot"]["cap"] = 2
            elif kind == "RNG":
                detail["phases"][3]["RNG_advancement_checks"] = 0
            elif kind == "bool_count":
                detail["runtime"]["device"] = False
            elif kind == "scope":
                detail["full_1900_cycle_certified"] = True

        for kind in (
            "missing",
            "source",
            "checkout_import",
            "metadata",
            "identity",
            "alias",
            "loader",
            "refund",
            "leak",
            "recovery",
            "epoch",
            "pending",
            "poison",
            "cap",
            "RNG",
            "bool_count",
            "scope",
        ):
            with self.subTest(kind=kind):
                detail, binding = complete_report()
                change(detail, kind)
                self.assertFalse(probe.report_passed(detail, binding))

    def test_timeout_and_uncertain_cleanup_remain_failure(self):
        safe = {
            "actual_exit": 0,
            "process_started": True,
            "direct_process_reaped": True,
            "cleanup_confirmed": True,
            "safe_to_continue": True,
            "remaining_owned_children_count": 0,
            "deadline_exceeded": False,
            "deadline_or_signal_exit": False,
        }
        self.assertTrue(utils.execution_safe(safe))
        for key, value in (
            ("actual_exit", 124),
            ("actual_exit", True),
            ("cleanup_confirmed", False),
            ("remaining_owned_children_count", 1),
            ("deadline_or_signal_exit", True),
        ):
            with self.subTest(key=key, value=value):
                result = dict(safe, **{key: value})
                self.assertFalse(utils.execution_safe(result))

    def test_source_guard_reads_bounded_regular_file_without_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(b"x=1\n")
            fingerprint = probe.source_fingerprint(path)
            self.assertEqual(fingerprint["bytes"], 4)
            empty = Path(directory) / "empty.py"
            empty.touch()
            with self.assertRaises(ValueError):
                probe.source_fingerprint(empty)

    def test_seal_is_exclusive_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            utils.sealed_json(path, {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                utils.sealed_json(path, {"status": "FAIL"})
            self.assertEqual(json.loads(path.read_text()), {"status": "PASS"})
            with self.assertRaises(ValueError):
                utils.sealed_json(Path(directory) / "bad.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
