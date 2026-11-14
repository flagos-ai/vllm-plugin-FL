# Copyright 2026 FlagOS Contributors
"""Offline contracts for the byte-exact, already executed roundtrip repro."""

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("repro_tensor_roundtrip.py")
EXECUTED_SHA256 = "c9c96bf53575811c0bdadaf8dc64e9dfe1ed602c228bd558f43af23320738d0f"


def load_script():
    spec = importlib.util.spec_from_file_location("repro_tensor_roundtrip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbid_accelerator_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_ptpu", "flag_gems", "vllm"}:
            raise AssertionError("An offline guard/help path imported " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_script_matches_executed_bytes():
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXECUTED_SHA256


def test_help_never_imports_accelerator_or_spawns(monkeypatch, capsys):
    forbid_accelerator_imports(monkeypatch)
    repro = load_script()
    monkeypatch.setattr(
        repro.mp, "get_context", lambda *args: pytest.fail("Help spawned a process")
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as result:
        repro.main()
    assert result.value.code == 0
    assert "--physical-device {0,2,4}" in capsys.readouterr().out


@pytest.mark.parametrize("mask", [None, "", "1", "2", "0,2"])
def test_wrong_or_multiple_masks_stop_before_accelerator_import(
    monkeypatch, tmp_path, mask
):
    forbid_accelerator_imports(monkeypatch)
    repro = load_script()
    if mask is None:
        monkeypatch.delenv("TANG_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("TANG_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        repro.mp,
        "get_context",
        lambda *args: pytest.fail("Rejected mask spawned a process"),
    )
    output = tmp_path / "unused"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--physical-device", "0", "--output-dir", str(output)],
    )
    with pytest.raises(SystemExit) as result:
        repro.main()
    assert result.value.code == 2
    assert not output.exists()


@pytest.fixture
def cpu_torch(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    return pytest.importorskip(
        "torch", reason="Full-value helper check needs optional CPU PyTorch"
    )


def test_full_value_check_detects_a_row_outside_sampled_indices(cpu_torch):
    torch = cpu_torch
    repro = load_script()
    expected = repro.cpu_pattern(torch, (64, 8), 0)
    assert expected.device.type == "cpu"
    actual = expected.clone()
    assert 17 not in torch.linspace(0, 63, 16).long().tolist()
    actual[17, 3] += 1
    gpu_metadata = {
        "shape": [64, 8],
        "stride": [8, 1],
        "element_size": 2,
        "data_ptr": 1000,
    }
    result = repro.compare_all(torch, actual, expected, gpu_metadata)
    assert result["compared_numel"] == 512
    assert result["mismatch_count"] == 1
    assert result["exact_values"] is False
    assert result["first_bad_row_col"] == [17, 3]
    assert result["first_bad_device_address"] == 1000 + 2 * (17 * 8 + 3)
    assert (
        repro.compare_all(torch, expected, expected, gpu_metadata)["exact_values"]
        is True
    )


def test_nan_is_recorded_and_view_offset_is_not_counted_twice(cpu_torch):
    torch = cpu_torch
    repro = load_script()
    expected = repro.cpu_pattern(torch, (3, 5), 79)
    actual = expected.clone()
    actual[1, 2] = float("nan")
    # data_ptr already includes any storage_offset, so use only logical strides.
    metadata = {
        "shape": [3, 5],
        "stride": [8, 1],
        "storage_offset": 7,
        "element_size": 2,
        "data_ptr": 1000,
    }
    result = repro.compare_all(torch, actual, expected, metadata)
    assert result["finite"] is False and result["exact_values"] is False
    assert result["mismatch_count"] == 1
    assert result["first_bad_device_address"] == 1020
    assert result["first_values_actual"][0] == "nan"
    json.dumps(result, allow_nan=False)
