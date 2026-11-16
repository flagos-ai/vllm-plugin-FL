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

from tests.e2e_tests.adaptation_gate.gate_contract import (
    QUALITY_CHECKS,
    SCENARIOS,
    append_loopback_no_proxy,
    build_server_args,
    gate_results_root,
    temporary_environment,
    validate_gate_documents,
)
from tests.run import _MODEL_E2E_TASKS, TestRunner
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


def test_contract_rejects_missing_scenario(tmp_path):
    _write_valid_documents(tmp_path)
    (tmp_path / "image_single.json").unlink()

    with pytest.raises(AssertionError, match="five original scenarios"):
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


def _fake_model_config(extra_args=None):
    serve = SimpleNamespace(
        extra_engine={"dtype": "bfloat16"},
        extra_args=list(extra_args or ["--reasoning-parser", "qwen3"]),
    )

    class Config:
        def __init__(self):
            self.serve = serve

        def serve_args(self, **overrides):
            assert overrides == {"dtype": "bfloat16"}
            return ["--max-model-len", "8192", "--dtype", "bfloat16"]

    return Config()


def test_build_server_args_preserves_config_and_allows_gate_images(tmp_path):
    image_dir = tmp_path / "images"

    args = build_server_args(_fake_model_config(), image_dir)

    assert args == [
        "--max-model-len",
        "8192",
        "--dtype",
        "bfloat16",
        "--reasoning-parser",
        "qwen3",
        "--allowed-local-media-path",
        str(image_dir.resolve()),
    ]


@pytest.mark.parametrize(
    "value",
    ["--allowed-local-media-path", "--allowed-local-media-path=/other/images"],
)
def test_build_server_args_rejects_duplicate_media_path(tmp_path, value):
    config = _fake_model_config([value])

    with pytest.raises(ValueError, match="must be supplied only"):
        build_server_args(config, tmp_path)


def test_runner_routes_adaptation_gate_to_unified_module(monkeypatch):
    config = SimpleNamespace(
        platform="demo",
        device="card",
        get_e2e_tests=lambda: E2eTestConfig(
            tests={"nightly": {"adaptation_gate": {"qwen3_6": ["27b_tp4_eager"]}}}
        ),
        should_skip_model=lambda _: False,
        get_e2e_variants=lambda: [],
    )
    monkeypatch.setattr(PlatformConfig, "load", lambda *args, **kwargs: config)

    runner = TestRunner(
        platform="demo",
        device="card",
        scope="e2e",
        task="adaptation_gate",
    )
    cases = runner.discover_tests()

    assert len(cases) == 1
    assert cases[0].task == "adaptation_gate"
    assert cases[0].pytest_path.endswith(
        "tests/e2e_tests/adaptation_gate/test_adaptation_gate.py"
    )
    assert cases[0].extra_env == {
        "FL_TEST_MODEL": "qwen3_6",
        "FL_TEST_CASE": "27b_tp4_eager",
    }
    assert "adaptation_gate" in _MODEL_E2E_TASKS


def test_matrix_declares_gate_timeout_and_tools_as_full_run_trigger():
    path = Path(__file__).parents[2] / ".github/scripts/generate_matrix.py"
    spec = importlib.util.spec_from_file_location("_gate_generate_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_TIMEOUT["adaptation_gate"] == 180
    assert "tools/adaptation-gate-cases/" in module._FULL_RUN_PREFIXES


def test_e2e_workflow_uploads_gate_documents():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/_e2e_test.yml"
    ).read_text()

    assert "test-results/gate/" in workflow
