# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only contracts for adaptation Gate orchestration."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

if importlib.util.find_spec("torch") is None:
    torch_stub = ModuleType("torch")
    torch_stub.__version__ = ""
    torch_stub.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    torch_stub.device = lambda value: value  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub

import tests.run as run_module
from tests.e2e_tests.adaptation_gate.gate_contract import (
    QUALITY_CHECKS,
    SCENARIOS,
    append_loopback_no_proxy,
    ensure_gate_server_alive,
    gate_results_root,
    resolve_request_timeout,
    temporary_environment,
    validate_gate_documents,
    validate_scenario_documents,
)
from tests.run import _MODEL_E2E_TASKS, TestCase, TestRunner
from tests.utils.platform_config import E2eTestConfig, PlatformConfig


def _write_valid_documents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for scenario, count in SCENARIOS:
        outputs = [
            {"passed": True, "checks": {name: True for name in QUALITY_CHECKS}}
            for _ in range(count)
        ]
        document = {
            "case": {"scenario": scenario},
            "input": [{} for _ in range(count)],
            "output": outputs,
            "summary": {
                "request_count": count,
                "passed_count": count,
                "failed_count": 0,
                "passed": True,
            },
        }
        (directory / f"{scenario}.json").write_text(json.dumps(document))


def _document(directory: Path, scenario: str) -> dict:
    return json.loads((directory / f"{scenario}.json").read_text())


def _replace_document(directory: Path, scenario: str, document: dict) -> None:
    (directory / f"{scenario}.json").write_text(json.dumps(document))


def test_contract_accepts_original_26_requests_and_260_checks(tmp_path):
    _write_valid_documents(tmp_path)

    assert validate_gate_documents(tmp_path) == {
        "logical_requests": 26,
        "checks": 260,
        "passed_checks": 260,
        "passed_requests": 26,
        "passed": True,
    }


def test_split_job_contracts_preserve_26_requests_and_260_checks(tmp_path):
    _write_valid_documents(tmp_path)
    serving = (("text_single", 1), ("image_single", 1))
    concurrent = (
        ("text_concurrent_8", 8),
        ("image_concurrent_8", 8),
        ("mixed_concurrent_8", 8),
    )
    serving_dir = tmp_path / "serving"
    concurrent_dir = tmp_path / "concurrent"
    serving_dir.mkdir()
    concurrent_dir.mkdir()
    for scenario, _ in serving:
        (serving_dir / f"{scenario}.json").write_text(
            (tmp_path / f"{scenario}.json").read_text()
        )
    for scenario, _ in concurrent:
        (concurrent_dir / f"{scenario}.json").write_text(
            (tmp_path / f"{scenario}.json").read_text()
        )

    serving_result = validate_scenario_documents(serving_dir, serving)
    concurrent_result = validate_scenario_documents(concurrent_dir, concurrent)

    assert serving_result["logical_requests"] == 2
    assert serving_result["checks"] == 20
    assert concurrent_result["logical_requests"] == 24
    assert concurrent_result["checks"] == 240


def test_contract_rejects_missing_scenario(tmp_path):
    _write_valid_documents(tmp_path)
    (tmp_path / "image_single.json").unlink()

    with pytest.raises(AssertionError, match="unexpected scenario set"):
        validate_gate_documents(tmp_path)


@pytest.mark.parametrize("field", ["input", "output"])
def test_contract_rejects_changed_request_count(tmp_path, field):
    _write_valid_documents(tmp_path)
    document = _document(tmp_path, "text_concurrent_8")
    document[field].pop()
    _replace_document(tmp_path, "text_concurrent_8", document)

    with pytest.raises(AssertionError, match=f"8 {field}"):
        validate_gate_documents(tmp_path)


def test_contract_rejects_missing_check(tmp_path):
    _write_valid_documents(tmp_path)
    document = _document(tmp_path, "text_single")
    document["output"][0]["checks"].pop("non_empty")
    _replace_document(tmp_path, "text_single", document)

    with pytest.raises(AssertionError, match="quality checks"):
        validate_gate_documents(tmp_path)


def test_contract_rejects_non_boolean_check(tmp_path):
    _write_valid_documents(tmp_path)
    document = _document(tmp_path, "text_single")
    document["output"][0]["checks"]["non_empty"] = 1
    _replace_document(tmp_path, "text_single", document)

    with pytest.raises(AssertionError, match="must all be bool"):
        validate_gate_documents(tmp_path)


def test_contract_rejects_259_of_260_passing_checks(tmp_path):
    _write_valid_documents(tmp_path)
    document = _document(tmp_path, "text_single")
    document["output"][0]["checks"]["non_empty"] = False
    document["output"][0]["passed"] = False
    document["summary"] = {
        "request_count": 1,
        "passed_count": 0,
        "failed_count": 1,
        "passed": False,
    }
    _replace_document(tmp_path, "text_single", document)

    with pytest.raises(AssertionError, match="quality contract failed"):
        validate_gate_documents(tmp_path)


def test_temporary_environment_exactly_restores_values(monkeypatch):
    monkeypatch.setenv("GATE_EXISTING", "before")
    monkeypatch.delenv("GATE_NEW", raising=False)

    with temporary_environment({"GATE_EXISTING": "during", "GATE_NEW": "value"}):
        assert os.environ["GATE_EXISTING"] == "during"
        assert os.environ["GATE_NEW"] == "value"

    assert os.environ["GATE_EXISTING"] == "before"
    assert "GATE_NEW" not in os.environ


def test_loopback_no_proxy_preserves_and_deduplicates_entries():
    value = append_loopback_no_proxy("example.com,127.0.0.1,example.com")

    assert value.split(",") == ["example.com", "127.0.0.1", "localhost"]


def test_request_timeout_prefers_platform_gate_override():
    environment = {
        "REQUEST_TIMEOUT": "300",
        "FL_ADAPTATION_GATE_REQUEST_TIMEOUT": "1800",
    }

    assert resolve_request_timeout(environment) == "1800"


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_request_timeout_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive number"):
        resolve_request_timeout({"FL_ADAPTATION_GATE_REQUEST_TIMEOUT": value})


def test_gate_progress_is_log_friendly_for_complete_contract(tmp_path):
    """The live total log line reports the real counters."""
    from tests.e2e_tests.adaptation_gate.gate_contract import format_gate_total_log

    _write_valid_documents(tmp_path)
    summary = validate_gate_documents(tmp_path)

    assert summary["passed"] is True
    assert format_gate_total_log("serving", summary) == (
        "[Gate] serving total: PASS; requests=26/26; checks=260/260"
    )


def test_gate_results_root_uses_safe_unique_components(tmp_path):
    eager = gate_results_root(
        tmp_path,
        platform="demo/vendor",
        device="card 0",
        model="qwen3_6",
        case="27b_tp4_eager",
    )
    graph = gate_results_root(
        tmp_path,
        platform="demo/vendor",
        device="card 0",
        model="qwen3_6",
        case="27b_tp4_graph",
    )

    assert eager != graph
    assert eager.relative_to(tmp_path.resolve()).parts == (
        "test-results",
        "gate",
        "demo-vendor",
        "card-0",
        "qwen3_6",
        "27b_tp4_eager",
    )


def test_runner_routes_standard_tasks_through_smoke_files(monkeypatch):
    config = SimpleNamespace(
        platform="demo",
        device="card",
        get_e2e_tests=lambda: E2eTestConfig(
            tests={
                "nightly": {
                    "serving": {"qwen3_6": ["27b_tp4_eager"]},
                    "concurrent": {"qwen3_6": ["27b_tp4_eager"]},
                }
            }
        ),
        should_skip_model=lambda _: False,
        get_e2e_variants=lambda: [],
    )
    monkeypatch.setattr(PlatformConfig, "load", lambda *args, **kwargs: config)

    runner = TestRunner(
        platform="demo",
        device="card",
        scope="e2e",
        task=None,
    )
    cases = runner.discover_tests()

    assert [case.task for case in cases] == ["serving", "concurrent"]
    assert cases[0].pytest_path.endswith("serving/test_serving_smoke.py")
    assert cases[1].pytest_path.endswith("concurrent/test_concurrent_smoke.py")
    assert all(
        case.extra_env
        == {
            "FL_TEST_MODEL": "qwen3_6",
            "FL_TEST_CASE": "27b_tp4_eager",
        }
        for case in cases
    )
    assert _MODEL_E2E_TASKS == ("inference", "serving", "concurrent")


@pytest.mark.parametrize(
    ("case_count", "expected_quiet"),
    [(1, False), (2, True)],
)
def test_e2e_runner_streams_single_case_live(monkeypatch, case_count, expected_quiet):
    runner = object.__new__(TestRunner)
    runner.config = SimpleNamespace(e2e_serial=False, platform="demo")
    runner.report = SimpleNamespace(results=[])
    quiet_values = []

    monkeypatch.setattr(run_module, "get_device_count", lambda: case_count)
    monkeypatch.setattr(run_module, "get_visible_device_env_var", lambda: "VISIBLE")
    monkeypatch.setattr(run_module, "device_cleanup", lambda *args, **kwargs: None)

    def fake_run_single(
        test_case,
        extra_env_override=None,
        _quiet=False,
        device_slots=None,
        device_ids=None,
    ):
        quiet_values.append(_quiet)
        return SimpleNamespace(name=test_case.name, passed=True, stdout="")

    monkeypatch.setattr(runner, "_run_single", fake_run_single)
    cases = [
        TestCase(
            name=f"concurrent/qwen3_6/case-{index}",
            pytest_path="unused",
            task="concurrent",
            num_devices=1,
        )
        for index in range(case_count)
    ]

    runner._run_e2e_parallel(cases)

    assert quiet_values == [expected_quiet] * case_count


def test_matrix_uses_standard_tasks_and_tools_as_full_run_trigger():
    path = Path(__file__).parents[2] / ".github/scripts/generate_matrix.py"
    spec = importlib.util.spec_from_file_location("_gate_generate_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "adaptation_gate" not in module.DEFAULT_TIMEOUT
    assert "tools/adaptation-gate-cases/" in module._FULL_RUN_PREFIXES


def test_matrix_groups_standard_task_cases_for_in_job_device_scheduling():
    path = Path(__file__).parents[2] / ".github/scripts/generate_matrix.py"
    spec = importlib.util.spec_from_file_location("_gate_shard_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "card": {
            "tests": {
                "e2e": {
                    "pr": {
                        "concurrent": {"qwen3_6": ["27b_tp4_eager", "27b_tp4_graph"]},
                        "inference": {"qwen3": ["4b_tp2", "8b_tp2"]},
                    }
                }
            }
        }
    }
    matrix = module.build_e2e_matrix(config, ["card"], [], ci_stage="pr")
    concurrent_entries = [entry for entry in matrix if entry["task"] == "concurrent"]
    inference_entries = [entry for entry in matrix if entry["task"] == "inference"]

    assert len(concurrent_entries) == 1
    assert len(json.loads(concurrent_entries[0]["cases"])) == 2
    assert len(inference_entries) == 1
    assert len(json.loads(inference_entries[0]["cases"])) == 2
    # One job per (task, device): both cases share the job and are scheduled
    # onto devices in-process by TestRunner's DeviceScheduler.
    assert {entry["device"] for entry in matrix} == {"card"}
    assert all(int(entry["timeout"]) > 0 for entry in matrix)


def test_e2e_workflow_uploads_gate_documents():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/_e2e_test.yml"
    ).read_text()

    assert "test-results/gate/" in workflow


@pytest.mark.parametrize(
    "case",
    sorted(
        path.stem
        for path in (Path(__file__).parents[1] / "models/qwen3_6").glob("*.yaml")
    ),
)
def test_qwen36_yamls_carry_explicit_inventory_without_verification(case):
    import yaml

    from tests.utils.model_config import ModelConfig

    raw = yaml.safe_load(
        (Path(__file__).parents[1] / f"models/qwen3_6/{case}.yaml").read_text()
    )
    assert "verification" not in raw, case
    assert len(raw["serve"]["chat_cases"]) == 26, case

    config = ModelConfig.load("qwen3_6", case)
    assert not hasattr(config, "verification"), case
    assert len(config.serve.chat_cases) == 26, case


def test_phase_results_use_distinct_paths(tmp_path):
    serving = gate_results_root(
        tmp_path,
        platform="demo",
        device="card",
        model="qwen3_6",
        case="27b_tp2_eager-serving",
    )
    concurrent = gate_results_root(
        tmp_path,
        platform="demo",
        device="card",
        model="qwen3_6",
        case="27b_tp2_eager-concurrent",
    )

    assert serving != concurrent


def test_dead_gate_server_fails_without_waiting_for_client_timeout():
    process = SimpleNamespace(poll=lambda: 42)
    server = SimpleNamespace(
        _process=process,
        preserve_failure_logs=lambda message: f"{message}: log",
    )

    with pytest.raises(RuntimeError, match="exited before the concurrent"):
        ensure_gate_server_alive(server, "concurrent")
