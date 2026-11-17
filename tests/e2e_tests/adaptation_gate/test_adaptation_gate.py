# Copyright (c) 2026 BAAI. All rights reserved.

"""Run the original 26-request adaptation Gate through the standard E2E stack."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from tests.e2e_tests.adaptation_gate.gate_contract import (
    SCENARIOS,
    append_loopback_no_proxy,
    build_server_args,
    format_gate_progress,
    gate_results_root,
    resolve_request_timeout,
    summarize_gate_progress,
    temporary_environment,
    validate_gate_documents,
)
from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.model_config import ModelConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CASE_FILE = _REPO_ROOT / "tools" / "adaptation-gate-cases" / "case_utils.py"

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")
_PLATFORM = os.environ.get("FL_TEST_PLATFORM", "")
_DEVICE = os.environ.get("FL_TEST_DEVICE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE, platform=_PLATFORM, device=_DEVICE)
if not os.path.exists(_CFG.model):
    pytest.fail(f"Model not found: {_CFG.model}", pytrace=False)


def _load_case_utils() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_adaptation_gate_cases", _CASE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adaptation Gate cases: {_CASE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CASE_UTILS = _load_case_utils()


@dataclass(frozen=True)
class GateServer:
    server: VllmServer
    document_dir: Path

    def ensure_alive(self) -> None:
        process = self.server._process
        if process is None or process.poll() is not None:
            pytest.fail(
                self.server.preserve_failure_logs(
                    "vLLM service exited before an adaptation Gate scenario"
                ),
                pytrace=False,
            )


@pytest.fixture(scope="module")
def gate_server():
    """Start one configured service and retain all five scenario documents."""
    serve = _CFG.serve
    image_dir = Path(_CASE_UTILS.IMAGE_DIR)
    extra_args = build_server_args(_CFG, image_dir)
    results_root = gate_results_root(
        _REPO_ROOT,
        platform=_PLATFORM,
        device=_DEVICE,
        model=_MODEL,
        case=_CASE,
    )

    with VllmServer(
        model=_CFG.model,
        tp_size=_CFG.engine.get("tensor_parallel_size", 1),
        api_key=serve.api_key,
        served_model_name=serve.served_model_name,
        max_retries=serve.startup_retries,
        extra_args=extra_args,
    ) as server:
        served_model_name = serve.request_model(_CFG.model)
        port = str(server.port)
        document_dir = (
            results_root
            / _CASE_UTILS.safe_slug(_CFG.model)
            / f"port-{_CASE_UTILS.safe_slug(port)}"
        )
        if any(document_dir.glob("*.json")):
            raise RuntimeError(f"Refusing to reuse stale Gate results: {document_dir}")
        updates = {
            "MODEL_PATH": _CFG.model,
            "SERVED_MODEL_NAME": served_model_name,
            "PORT": port,
            "BASE_URL": server.base_url,
            "API_KEY": serve.api_key or "EMPTY",
            "RESULTS_DIR": str(results_root),
            "REQUEST_TIMEOUT": resolve_request_timeout(os.environ),
            "NO_PROXY": append_loopback_no_proxy(os.environ.get("NO_PROXY")),
            "no_proxy": append_loopback_no_proxy(os.environ.get("no_proxy")),
        }
        with temporary_environment(updates):
            try:
                yield GateServer(server=server, document_dir=document_dir)
            finally:
                print(
                    format_gate_progress(
                        "total", summarize_gate_progress(document_dir)
                    ),
                    flush=True,
                )
                with server.log_on_failure(
                    "Adaptation Gate 26-request contract validation failed"
                ):
                    validate_gate_documents(document_dir)


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", [name for name, _ in SCENARIOS])
def test_adaptation_gate_scenario(scenario: str, gate_server: GateServer) -> None:
    """Run one original scenario while sharing the model service."""
    gate_server.ensure_alive()
    with gate_server.server.log_on_failure(
        f"Adaptation Gate scenario '{scenario}' failed"
    ):
        try:
            _CASE_UTILS.run_case(scenario)
        finally:
            print(
                format_gate_progress(
                    scenario,
                    summarize_gate_progress(gate_server.document_dir, scenario),
                ),
                flush=True,
            )
