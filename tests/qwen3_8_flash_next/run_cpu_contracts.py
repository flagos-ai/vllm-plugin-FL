#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce Qwen3.8's 76 CPU contracts without accelerator access.

Run inside the matching validation environment:
    python tests/qwen3_8_flash_next/run_cpu_contracts.py --log-dir /tmp/qwen38-cpu

The 56 model/MTP contracts and 20 dispatch contracts run in separate processes.
The dispatch subprocess loads production providers with only framework/package
capability imports stubbed; it does not validate device kernels or imports.
The model/MTP subprocess retains the real native metadata tests and the
repository's existing CPU declaration harnesses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def cpu_environment():
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        MACA_VISIBLE_DEVICES="",
        VLLM_PLUGINS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    return env


def install_dispatch_import_stubs(repo):
    # Import Torch before installing minimal packages so Torch inspection sees
    # real module metadata. No accelerator is visible in this subprocess.
    import torch  # noqa: F401

    def package(name):
        if name in sys.modules:
            return sys.modules[name]
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, key = name.rsplit(".", 1)
            setattr(package(parent), key, module)
        return module

    def load(name, relative):
        parent, key = name.rsplit(".", 1)
        package(parent)
        spec = importlib.util.spec_from_file_location(name, repo / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(sys.modules[parent], key, module)

    capability = package("vllm_fl.dispatch.capabilities")
    capability.get_flaggems_vllm_operator = lambda name: None
    envs = package("vllm_fl.envs")

    def environment_value(name):
        if name == "VLLM_FL_USE_FLAGGEMS_VLLM":
            return os.environ.get(name, "0") == "1"
        raise AttributeError(name)

    envs.__getattr__ = environment_value
    load("vllm_fl.dispatch.types", "vllm_fl/dispatch/types.py")
    load(
        "vllm_fl.dispatch.backends.flaggems.impl.qwen3_8_flash_next",
        "vllm_fl/dispatch/backends/flaggems/impl/qwen3_8_flash_next.py",
    )
    load(
        "vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next",
        "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/__init__.py",
    )


class ResultRecorder:
    def __init__(self, path):
        self.path = path
        self.passed = set()
        self.failed = set()
        self.skipped = set()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)
        if report.failed:
            self.failed.add(report.nodeid)
        if report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        result = {
            "collected": session.testscollected,
            "passed": len(self.passed),
            "failed": sorted(self.failed),
            "skipped": sorted(self.skipped),
            "exit_code": int(exitstatus),
        }
        self.path.write_text(json.dumps(result, indent=2) + "\n")


def run_suite(args):
    os.environ.update(cpu_environment())
    os.chdir(args.repo)
    if args.suite == "dispatch":
        install_dispatch_import_stubs(args.repo)
        options = [
            "--confcutdir=tests/unit_tests/dispatch",
            "-q",
            "tests/unit_tests/dispatch/test_qwen38_fgv.py",
        ]
    else:
        options = [
            "--confcutdir=tests/qwen3_8_flash_next",
            "-q",
            *[
                str(p.relative_to(args.repo))
                for p in sorted(
                    (args.repo / "tests/unit_tests/models").glob("test_qwen38_*.py")
                )
            ],
            "tests/qwen3_8_flash_next/test_lazy_registration.py",
            "tests/qwen3_8_flash_next/test_mtp_contract.py",
            "tests/qwen3_8_flash_next/test_mtp_forward.py",
            "tests/qwen3_8_flash_next/test_mtp_multi_group.py",
            "tests/qwen3_8_flash_next/test_qsa_cache_binding.py",
        ]
    import pytest

    result_file = args.log_dir / f"qwen38-final-cpu-{args.suite}.json"
    print(f"CPU suite={args.suite}; pytest arguments={options}", flush=True)
    return int(pytest.main(options, plugins=[ResultRecorder(result_file)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("contracts", "dispatch"))
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.suite:
        return run_suite(args)

    print(f"Python: {sys.version.split()[0]}; repository: {args.repo}", flush=True)
    print(
        "CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='', VLLM_PLUGINS=''", flush=True
    )
    summaries = {}
    success = True
    for suite, expected in (("contracts", 56), ("dispatch", 20)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--repo",
            str(args.repo),
            "--log-dir",
            str(args.log_dir),
            "--suite",
            suite,
        ]
        result_path = args.log_dir / f"qwen38-final-cpu-{suite}.json"
        result_path.unlink(missing_ok=True)
        result = subprocess.run(command, cwd=args.repo, env=cpu_environment())
        summary = json.loads(result_path.read_text()) if result_path.exists() else {}
        summaries[suite] = summary
        correct = (
            result.returncode == 0
            and summary.get("collected") == expected
            and summary.get("passed") == expected
            and not summary.get("failed")
            and not summary.get("skipped")
        )
        success = success and correct
        print(
            f"RESULT {suite}: {summary}; expected={expected}; valid={correct}",
            flush=True,
        )
    total = sum(value.get("passed", 0) for value in summaries.values())
    final = {"passed": total, "expected": 76, "success": success, "suites": summaries}
    (args.log_dir / "qwen38-final-cpu-summary.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    print(f"FINAL CPU CONTRACTS: {total}/76 passed; success={success}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
