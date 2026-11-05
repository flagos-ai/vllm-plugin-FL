"""CPU-only contracts for the byte-frozen, executed TP1 V2 smoke script."""

import builtins
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("smoke_single_card_v2.py")
EXECUTED_SHA256 = "e0bb0b35491c6f448622cde983350900512a3f6bb2212fea677ea44ae6cd2671"


def load():
    spec = importlib.util.spec_from_file_location("single_card_smoke_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proof(mode, replays=(0, 0)):
    graph = mode == "graph"
    return [
        {
            "rank": 0,
            "runner": "SunriseV2ModelRunner",
            "runner_v2": True,
            "tensor_parallel_size": 1,
            "enforce_eager": not graph,
            "async_scheduling": True,
            "graph_mode": "FULL_DECODE_ONLY" if graph else "NONE",
            "capture_sizes": [1, 2] if graph else [],
            "graphs": [
                {
                    "descriptor": str(index + 1),
                    "native_type": "PTPUGraph",
                    "replays": count,
                }
                for index, count in enumerate(replays)
            ]
            if graph
            else [],
        }
    ]


def forbid_gpu_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "torch_ptpu",
            "vllm",
            "vllm_fl",
            "triton",
            "flag_gems",
        }:
            pytest.fail("Offline guard/help imported accelerator package: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_source_matches_executed_hash():
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXECUTED_SHA256


def test_valid_proof_accepts_eager_or_actual_postready_replays():
    smoke = load()
    smoke.validate_proof("eager", proof("eager"), proof("eager"))
    smoke.validate_proof("graph", proof("graph"), proof("graph", (5, 3)))


@pytest.mark.parametrize("counts", [(0, 0), (0, 3), (5, 0)])
def test_graph_rejects_capture_only_or_one_unused_graph(counts):
    with pytest.raises(AssertionError):
        load().validate_proof("graph", proof("graph"), proof("graph", counts))


@pytest.mark.parametrize("phase", ["before", "after"])
def test_wrong_runner_is_rejected(phase):
    before, after = proof("graph"), proof("graph", (5, 3))
    (before if phase == "before" else after)[0]["runner"] = "GPUModelRunner"
    with pytest.raises(AssertionError):
        load().validate_proof("graph", before, after)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_eager_with_any_captured_graph_is_rejected(phase):
    before, after = proof("eager"), proof("eager")
    (before if phase == "before" else after)[0]["graphs"] = copy.deepcopy(
        proof("graph")[0]["graphs"]
    )
    with pytest.raises(AssertionError):
        load().validate_proof("eager", before, after)


def test_help_imports_no_accelerator_package(monkeypatch, capsys):
    forbid_gpu_imports(monkeypatch)
    with pytest.raises(SystemExit) as result:
        load().main(["--help"])
    assert result.value.code == 0
    assert "--mode {eager,graph}" in capsys.readouterr().out


@pytest.mark.parametrize("mask", [None, "0", "0,2", "4,5"])
def test_wrong_or_missing_mask_stops_before_gpu_import_and_output_creation(
    monkeypatch, tmp_path, mask
):
    forbid_gpu_imports(monkeypatch)
    if mask is None:
        monkeypatch.delenv("TANG_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("TANG_VISIBLE_DEVICES", mask)
    output = tmp_path / "new-output"
    with pytest.raises(SystemExit) as result:
        load().main(["--mode", "eager", "--output-dir", str(output)])
    assert result.value.code == 2
    assert not output.exists()


def test_existing_output_stops_before_gpu_import_without_overwrite(
    monkeypatch, tmp_path
):
    forbid_gpu_imports(monkeypatch)
    monkeypatch.setenv("TANG_VISIBLE_DEVICES", "4")
    output = tmp_path / "existing-output"
    output.mkdir()
    receipt = output / "receipt.json"
    receipt.write_text('{"historical":true}\n', encoding="utf-8")
    original = receipt.read_bytes()
    with pytest.raises(SystemExit) as result:
        load().main(["--mode", "graph", "--output-dir", str(output)])
    assert result.value.code == 2
    assert receipt.read_bytes() == original
