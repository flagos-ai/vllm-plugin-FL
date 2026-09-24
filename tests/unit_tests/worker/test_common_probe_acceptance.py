"""Acceptance must reject shared state bugs and checkout-shadowed wheels."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[2] / "e2e_tests/inference/common_metadata_probe.py"
)
spec = importlib.util.spec_from_file_location("common_metadata_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_shared_cross_batch_drift_fails_even_when_policies_agree():
    output = {"prompt": [1, 2], "tokens": [11], "logprobs": [{"11": -0.1}]}
    run = {"results": [[output], [copy.deepcopy(output)]]}
    runs = {mode: copy.deepcopy(run) for mode in ("stock", "eager", "graph")}
    assert probe.compare_runs(runs, 0) == 0
    for result in runs.values():
        result["results"][1][0]["tokens"] = [99]
    with pytest.raises(AssertionError, match="request reuse"):
        probe.compare_runs(runs, 0)


def test_worker_identity_rejects_checkout_and_changed_runtime(tmp_path):
    installed = tmp_path / "installed"
    name = probe.RUNTIME_MODULES[0]
    relative = name.replace(".", "/") + ".py"
    identity = {
        "package_path": str(installed / "vllm_fl/__init__.py"),
        "files": {name: {"path": str(installed / relative), "sha256": "tested"}},
    }
    probe.verify_identity(identity, installed, {relative: "tested"})
    identity["package_path"] = str(tmp_path / "checkout/vllm_fl/__init__.py")
    with pytest.raises(AssertionError):
        probe.verify_identity(identity, installed, {relative: "tested"})
    identity["package_path"] = str(installed / "vllm_fl/__init__.py")
    with pytest.raises(AssertionError):
        probe.verify_identity(identity, installed, {relative: "different wheel"})


def test_fp16_combination_still_rejects_shared_logprob_drift():
    output = {"prompt": [1, 2], "tokens": [11], "logprobs": [{"11": -0.1}]}
    run = {"results": [[output], [copy.deepcopy(output)]]}
    runs = {mode: copy.deepcopy(run) for mode in ("stock", "eager", "graph")}
    for result in runs.values():
        result["results"][1][0]["logprobs"][0]["11"] += 0.002
    with pytest.raises(AssertionError, match="request reuse"):
        probe.compare_runs(runs, 2**-10)
