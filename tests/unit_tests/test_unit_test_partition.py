# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise unit-pass discovery and real pytest marker selection on the CPU."""

import itertools
import os
import subprocess
import sys

import pytest

from tests import run as test_runner
from tests.utils.platform_config import TestFilter as UnitFilter


@pytest.fixture
def runner(monkeypatch):
    runner = test_runner.TestRunner(platform="metax", scope="unit", dry_run=True)

    def unexpected_device_probe():
        pytest.fail("Unit discovery must not query accelerator availability")

    monkeypatch.setattr(test_runner, "get_device_count", unexpected_device_probe)
    return runner


def _selection_args(case):
    """Use the runner's selectors without requiring its reporting plugins."""
    args = []
    for index, value in enumerate(case.extra_args):
        if value in ("-m", "-k", "--ignore"):
            args.extend((value, case.extra_args[index + 1]))
    return args


@pytest.fixture
def marker_suite(tmp_path):
    """Create all eight marker combinations, including filtered CPU/GPU tests."""
    markers = ("gpu", "multi_gpu", "serial")
    config = tmp_path / "pytest.ini"
    config.write_text(
        "[pytest]\nmarkers =\n" + "".join(f"    {name}\n" for name in markers),
        encoding="utf-8",
    )
    nodes = {}
    for relative, prefix in (
        ("test_selected.py", "kept"),
        ("test_also_selected.py", "other"),
        ("test_unselected.py", "filtered"),
        ("excluded/test_excluded.py", "kept_excluded"),
    ):
        path = tmp_path / "tests" / "unit_tests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = ["import pytest\n"]
        for flags in itertools.product((False, True), repeat=len(markers)):
            selected = tuple(name for name, enabled in zip(markers, flags) if enabled)
            name = f"test_{prefix}_{'_'.join(selected) or 'cpu'}"
            source.extend(f"@pytest.mark.{marker}\n" for marker in selected)
            source.append(f"def {name}():\n    pass\n\n")
            node = f"tests/unit_tests/{relative}::{name}"
            nodes[node] = selected
        path.write_text("".join(source), encoding="utf-8")
    return tmp_path, nodes


def _collect(case, root):
    env = os.environ.copy()
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        PYTEST_PLUGINS="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "--noconftest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(root / "pytest.ini"),
            case.pytest_path,
            *_selection_args(case),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        line.replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.startswith("tests") and "::" in line
    }


@pytest.mark.parametrize("filtered", (False, True))
def test_unit_passes_partition_real_pytest_markers(
    runner, monkeypatch, marker_suite, filtered
):
    root, nodes = marker_suite
    unit_filter = (
        UnitFilter(include=["kept", "other"], exclude=["excluded/"])
        if filtered
        else UnitFilter()
    )
    monkeypatch.setattr(runner.config, "get_unit_filter", lambda: unit_filter)
    parallel_case, serial_case = runner._discover_unit_tests()
    parallel = _collect(parallel_case, root)
    serial = _collect(serial_case, root)
    eligible = {
        node: markers
        for node, markers in nodes.items()
        if not filtered
        or (
            "/excluded/" not in node and ("test_kept_" in node or "test_other_" in node)
        )
    }

    assert parallel.isdisjoint(serial)
    assert parallel | serial == eligible.keys()
    assert parallel == {node for node, markers in eligible.items() if not markers}
    assert serial == {node for node, markers in eligible.items() if markers}
    # In particular, gpu + multi_gpu overlap still appears exactly once.
    assert any(
        "gpu" in eligible[node] and "multi_gpu" in eligible[node] for node in serial
    )


@pytest.mark.parametrize("cpu_count, workers", ((64, 16), (8, 8), (None, 1)))
def test_unit_passes_keep_worker_and_reporting_contracts(
    runner, monkeypatch, cpu_count, workers
):
    monkeypatch.setattr(test_runner.os, "cpu_count", lambda: cpu_count)
    parallel, serial = runner._discover_unit_tests()

    assert parallel.extra_args[parallel.extra_args.index("-n") + 1] == str(workers)
    assert "-n" not in serial.extra_args
    assert "--cov-append" not in parallel.extra_args
    assert "--cov-append" in serial.extra_args
    assert [case.name for case in (parallel, serial)] == [
        "unit (metax)",
        "unit-serial (metax)",
    ]
    for case, report in (
        (parallel, "report-metax.json"),
        (serial, "report-metax-serial.json"),
    ):
        assert case.pytest_path == "tests/unit_tests/"
        assert case.task == "unit"
        assert case.num_devices == 0
        assert case.extra_env == {}
        for argument in (
            "--cov=vllm_fl",
            "--cov-report=term-missing",
            "--cov-report=json:coverage-metax.json",
            "--json-report",
            f"--json-report-file={report}",
        ):
            assert argument in case.extra_args
