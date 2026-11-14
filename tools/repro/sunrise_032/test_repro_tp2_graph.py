# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts: importing/checking this repro must never initialize a GPU."""

import argparse
import ast
import builtins
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("repro_tp2_graph.py")


def load_repro():
    spec = importlib.util.spec_from_file_location(
        "sunrise_graph_repro_contract", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repro():
    return load_repro()


def test_import_does_not_import_accelerator_packages(monkeypatch):
    original = builtins.__import__
    forbidden = {"torch", "torch_ptpu", "vllm", "vllm_fl", "triton", "flagcx"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"hardware import during module import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    load_repro()


def test_help_works_without_any_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--visible-devices" in result.stdout
    assert "NOT been hardware-retested" in " ".join(result.stdout.split())


def test_device_selection_is_required_before_hardware_import():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "--visible-devices" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("mask", ["", "0", "0,1,2", "1,1", "-1,2", "auto,1"])
def test_invalid_device_pairs_are_rejected(repro, mask):
    with pytest.raises(argparse.ArgumentTypeError):
        repro.visible_pair(mask)


def test_explicit_pair_order_is_retained(repro):
    assert repro.visible_pair(" 5, 4 ") == "5,4"


def test_environment_validation_is_pure_and_preserves_existing_mask(repro):
    args = repro.build_parser().parse_args(["--visible-devices", "5,4"])
    original = {"TANG_VISIBLE_DEVICES": "5,4", "KEEP_USER_VALUE": "unchanged"}
    updates = repro.environment_updates(args, original)
    assert original == {"TANG_VISIBLE_DEVICES": "5,4", "KEEP_USER_VALUE": "unchanged"}
    assert "TANG_VISIBLE_DEVICES" not in updates
    assert updates["TA_CLOSE_GRAPH_MEMORY"] == "1"
    assert updates["PCCL_PROTO"] == "Simple"


def test_selected_pair_is_used_when_no_mask_is_set(repro):
    args = repro.build_parser().parse_args(["--visible-devices", "4,5"])
    assert repro.environment_updates(args, {})["TANG_VISIBLE_DEVICES"] == "4,5"


def test_conflicting_device_selection_fails_before_hardware_import():
    env = os.environ.copy()
    env["TANG_VISIBLE_DEVICES"] = "0,1"
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--visible-devices", "4,5"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert result.returncode == 2
    assert "conflicts with TANG_VISIBLE_DEVICES" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "name,value",
    [("TA_CLOSE_GRAPH_MEMORY", "0"), ("PCCL_PROTO", "other"), ("GEMS_VENDOR", "other")],
)
def test_conflicting_runtime_settings_are_not_overwritten(repro, name, value):
    args = repro.build_parser().parse_args(["--visible-devices", "4,5"])
    original = {name: value}
    with pytest.raises(ValueError, match=name):
        repro.environment_updates(args, original)
    assert original == {name: value}


def test_historical_unset_mode_neither_sets_nor_removes_user_variable(repro):
    args = repro.build_parser().parse_args(
        ["--visible-devices", "4,5", "--ta-close-graph-memory", "unset"]
    )
    assert "TA_CLOSE_GRAPH_MEMORY" not in repro.environment_updates(args, {})
    with pytest.raises(ValueError, match="requires TA_CLOSE_GRAPH_MEMORY to be absent"):
        repro.environment_updates(args, {"TA_CLOSE_GRAPH_MEMORY": "1"})


def test_same_parent_pid_always_gets_new_store_without_creating_it(repro, tmp_path):
    first, second = repro.new_rendezvous(tmp_path), repro.new_rendezvous(tmp_path)
    assert first != second
    assert str(os.getpid()) in first.name
    assert first.is_absolute()
    assert not first.exists() and not second.exists()


def test_existing_store_is_preserved_and_not_retried(repro, tmp_path, monkeypatch):
    fixed = type("FixedUUID", (), {"hex": "fixed"})()
    monkeypatch.setattr(repro.uuid, "uuid4", lambda: fixed)
    store = tmp_path / f"sunrise_tp2_graph_{os.getpid()}_fixed"
    store.write_text("stale evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        repro.new_rendezvous(tmp_path)
    assert store.read_text(encoding="utf-8") == "stale evidence"


def test_async_sequence_preserves_the_unsynchronized_failure_path():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 10))
    worker = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"
    )
    request_loop = next(
        n
        for n in ast.walk(worker)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and any(isinstance(t, ast.Name) and t.id == "iteration" for t in n.target.elts)
    )
    calls = [n.func for n in ast.walk(request_loop) if isinstance(n, ast.Call)]
    forbidden = {"synchronize", "barrier", "wait_stream", "wait_event", "sleep"}
    assert not any(
        isinstance(call, ast.Attribute) and call.attr in forbidden for call in calls
    )
    assert any(
        isinstance(call, ast.Attribute) and call.attr == "replay" for call in calls
    )
    assert any(
        isinstance(call, ast.Attribute) and call.attr == "all_gather" for call in calls
    )


MODEL_SCRIPT = SCRIPT.with_name("repro_v2_model_graph.py")


def test_model_help_works_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", str(MODEL_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout and "--output-dir" in result.stdout
    assert "not GPU-retested" in result.stdout


def test_model_requires_explicit_model_and_output_directory():
    result = subprocess.run(
        [sys.executable, "-S", str(MODEL_SCRIPT), "--visible-devices", "4,5"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "--model" in result.stderr and "--output-dir" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_model_mask_conflict_fails_before_import_or_output_creation(tmp_path):
    env = os.environ.copy()
    env["TANG_VISIBLE_DEVICES"] = "0,1"
    output = tmp_path / "must-not-create"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(MODEL_SCRIPT),
            "--visible-devices",
            "4,5",
            "--model",
            "/explicit/model",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert result.returncode == 2
    assert "conflicts with TANG_VISIBLE_DEVICES" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.exists()


def test_model_preserves_original_batch_switching_without_replay_proxy():
    tree = ast.parse(MODEL_SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 10))
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    checks = next(
        n.value
        for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "checks" for t in n.targets)
    )
    assert [len(batch) for batch in ast.literal_eval(checks)] == [2, 1, 2]
    assert not any(
        isinstance(n, ast.ClassDef) and n.name == "ReplayCounter"
        for n in ast.walk(tree)
    )
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"replay", "synchronize"}
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "name,value", [("VLLM_PLUGINS", "other"), ("VLLM_USE_V2_MODEL_RUNNER", "0")]
)
def test_plugin_and_v2_conflicts_fail_before_any_hardware_import(name, value):
    env = os.environ.copy()
    for key in (
        "TANG_VISIBLE_DEVICES",
        "TA_CLOSE_GRAPH_MEMORY",
        "PCCL_PROTO",
        "GEMS_VENDOR",
        "VLLM_PLUGINS",
        "VLLM_USE_V2_MODEL_RUNNER",
    ):
        env.pop(key, None)
    env[name] = value
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--visible-devices", "4,5"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert result.returncode == 2
    assert name in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_existing_plugin_list_including_fl_is_retained(repro):
    args = repro.build_parser().parse_args(["--visible-devices", "4,5"])
    updates = repro.environment_updates(args, {"VLLM_PLUGINS": "other, fl"})
    assert "VLLM_PLUGINS" not in updates


def test_model_missing_local_config_fails_before_gpu_import_or_download(tmp_path):
    env = os.environ.copy()
    for key in (
        "TANG_VISIBLE_DEVICES",
        "TA_CLOSE_GRAPH_MEMORY",
        "PCCL_PROTO",
        "GEMS_VENDOR",
        "VLLM_PLUGINS",
        "VLLM_USE_V2_MODEL_RUNNER",
    ):
        env.pop(key, None)
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(MODEL_SCRIPT),
            "--visible-devices",
            "4,5",
            "--model",
            str(tmp_path / "missing-model"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert result.returncode == 2
    assert "local model directory containing config.json" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.exists()
