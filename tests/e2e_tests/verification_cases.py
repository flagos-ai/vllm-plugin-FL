# Copyright (c) 2026 BAAI. All rights reserved.

"""Adapters that run external verification case packs in standard E2E jobs."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from tests.e2e_tests.adaptation_gate.gate_contract import (
    append_loopback_no_proxy,
    gate_results_root,
    resolve_request_timeout,
    temporary_environment,
)

ADAPTATION_GATE_SOURCE = "tools/adaptation-gate-cases"
ADAPTATION_GATE_SCENARIOS = {
    "text_single": 1,
    "text_concurrent_8": 8,
    "image_single": 1,
    "image_concurrent_8": 8,
    "mixed_concurrent_8": 8,
}


def validate_case_selection(source: str, scenarios: Sequence[str]) -> None:
    """Fail early when a model YAML references an unknown pack or scenario."""
    if source != ADAPTATION_GATE_SOURCE:
        raise ValueError(f"Unsupported verification case source: {source!r}")
    if not scenarios:
        raise ValueError("Verification scenarios must not be empty")
    unknown = sorted(set(scenarios) - ADAPTATION_GATE_SCENARIOS.keys())
    if unknown:
        raise ValueError(f"Unknown adaptation Gate scenarios: {unknown}")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Verification scenarios must not contain duplicates")


def load_case_pack(repo_root: Path, source: str) -> ModuleType:
    """Load a supported tools case pack without copying its request data."""
    if source != ADAPTATION_GATE_SOURCE:
        raise ValueError(f"Unsupported verification case source: {source!r}")
    case_file = repo_root / source / "case_utils.py"
    spec = importlib.util.spec_from_file_location("_verification_case_pack", case_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load verification cases: {case_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verification_results_root(
    repo_root: Path,
    *,
    phase: str,
    platform: str,
    device: str,
    model: str,
    case: str,
) -> Path:
    """Keep serving and concurrent documents separate for one model case."""
    return gate_results_root(
        repo_root,
        platform=platform,
        device=device,
        model=model,
        case=f"{case}-{phase}",
    )


def ensure_server_alive(server, phase: str) -> None:
    """Fail immediately when a shared verification server has exited."""
    process = server._process
    if process is None or process.poll() is not None:
        details = server.preserve_failure_logs(
            f"vLLM service exited before the {phase} verification scenario"
        )
        raise RuntimeError(details)


@contextmanager
def case_pack_environment(
    *,
    case_utils: ModuleType,
    config,
    server,
    results_root: Path,
) -> Iterator[Path]:
    """Expose a standard VllmServer through the tools case-pack env contract."""
    served_model_name = config.serve.request_model(config.model)
    port = str(server.port)
    document_dir = (
        results_root
        / case_utils.safe_slug(config.model)
        / f"port-{case_utils.safe_slug(port)}"
    )
    if any(document_dir.glob("*.json")):
        raise RuntimeError(
            f"Refusing to reuse stale verification results: {document_dir}"
        )
    updates = {
        "MODEL_PATH": config.model,
        "SERVED_MODEL_NAME": served_model_name,
        "PORT": port,
        "BASE_URL": server.base_url,
        "API_KEY": config.serve.api_key or "EMPTY",
        "RESULTS_DIR": str(results_root),
        "REQUEST_TIMEOUT": resolve_request_timeout(os.environ),
        "NO_PROXY": append_loopback_no_proxy(os.environ.get("NO_PROXY")),
        "no_proxy": append_loopback_no_proxy(os.environ.get("no_proxy")),
    }
    with temporary_environment(updates):
        yield document_dir
