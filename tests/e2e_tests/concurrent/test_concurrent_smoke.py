# Copyright (c) 2025 BAAI. All rights reserved.

"""
Concurrent generation smoke test driven by model YAML configs.

Adapted from the sglang-plugin-FL concurrent smoke test. Two execution
modes, chosen from the YAML:

1. Explicit ``task: concurrent`` chat cases run as real HTTP concurrency:
   one shared server, one scenario per pytest parameter, one
   ``asyncio.gather`` of eight requests per scenario, each response graded
   with the ten quality checks and written to ``test-results/gate/``.
2. Otherwise the classic offline smoke runs: N requests are submitted to a
   single engine in one ``LLM.generate`` call, so the scheduler runs them
   through continuous batching concurrently, then every output is
   validated. The ``concurrent`` block in the model YAML is optional —
   defaults keep the CI smoke small (8 requests, 64 tokens each).

This test file is driven by environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3_6``)
- ``FL_TEST_CASE``:  Case name within the family (e.g. ``27b_tp2_eager``)
- ``FL_TEST_PLATFORM``/``FL_TEST_DEVICE``: Apply platform device overrides

Offline-mode overrides (optional): ``FL_CONCURRENT_N``, ``FL_CONCURRENT_MAX_TOKENS``.
"""

import contextlib
import os
import time
from pathlib import Path

import pytest

from tests.e2e_tests.adaptation_gate.gate_contract import (
    append_loopback_no_proxy,
    ensure_clean_gate_directory,
    ensure_gate_server_alive,
    format_gate_scenario_log,
    format_gate_total_log,
    gate_failure_details,
    gate_results_root,
    gate_run_id,
    resolve_request_timeout,
    run_concurrent_scenario,
    temporary_environment,
    validate_scenario_documents,
)
from tests.e2e_tests.chat_cases import (
    assemble_gate_extra_args,
    gate_entry,
    group_concurrent_cases,
    validate_chat_case_tasks,
    validate_gate_case,
)
from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.device_utils import get_visible_device_env_var
from tests.utils.model_config import ModelConfig
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# ---------------------------------------------------------------------------
# Load config from environment (injected by run.py)
# ---------------------------------------------------------------------------

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
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GATE_RUN_ID = gate_run_id(os.environ)

# Execution mode: explicit task cases take the quality-gate HTTP path;
# models without them keep the classic offline LLM.generate smoke.
validate_chat_case_tasks(_CFG.serve.chat_cases)
_CONCURRENT_CASES = [
    case for case in _CFG.serve.chat_cases if case.get("task") == "concurrent"
]
if _CONCURRENT_CASES:
    for _gate_case in _CONCURRENT_CASES:
        validate_gate_case(_gate_case, "concurrent")
_SCENARIO_GROUPS = (
    group_concurrent_cases(_CFG.serve.chat_cases) if _CONCURRENT_CASES else {}
)
_SCENARIOS = tuple(_SCENARIO_GROUPS)

if not os.path.exists(_CFG.model):
    pytest.fail(
        f"Model not found: {_CFG.model}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Prompt pool and sampling (offline mode)
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_CASES = [
    {
        "text": "How many states are there in the United States?",
        "expected": ["50"],
    },
    {"text": "The capital of France is", "expected": ["paris"]},
    {
        "text": "What is the largest planet in the solar system?",
        "expected": ["jupiter"],
    },
    {"text": "Who wrote the play Romeo and Juliet?", "expected": ["shakespeare"]},
    {"text": "What is 17 multiplied by 13?", "expected": ["221"]},
    {
        "text": "Name the three primary colors of light.",
        "expected": ["red", "green", "blue"],
    },
    {"text": "What year did World War II end?", "expected": ["1945"]},
    {
        "text": "Explain the concept of gravity in one sentence.",
        "expected": ["mass", "force", "attract"],
    },
]


def _normalize_text_cases(raw_cases: list) -> list[dict]:
    cases = []
    for item in raw_cases:
        if isinstance(item, str):
            cases.append({"text": item, "expected": []})
            continue
        expected = item.get("expected", [])
        if isinstance(expected, str):
            expected = [expected]
        cases.append({"text": item["text"], "expected": expected})
    return cases


_TEXT_CASES = _normalize_text_cases(_CFG.concurrent.text_prompts or _DEFAULT_TEXT_CASES)
CONCURRENT_N = int(os.environ.get("FL_CONCURRENT_N", str(_CFG.concurrent.concurrent_n)))
_SAMPLING = {"max_tokens": 64, "temperature": 0.0}
_SAMPLING.update(_CFG.concurrent.text_sampling)
_SAMPLING["max_tokens"] = int(
    os.environ.get("FL_CONCURRENT_MAX_TOKENS", _SAMPLING["max_tokens"])
)

_REQUEST_MODEL = _CFG.serve.request_model(_CFG.model)


def _prepare_cleanup_for_platform() -> None:
    """Apply test-scope cleanup compatibility hooks."""
    if _PLATFORM != "ascend":
        return

    import torch

    npu = getattr(torch, "npu", None)
    accelerator = getattr(torch, "accelerator", None)
    if npu is not None and accelerator is not None and hasattr(npu, "empty_cache"):
        accelerator.empty_cache = npu.empty_cache


def _report(n_req: int, elapsed: float, total_tokens: int) -> None:
    print(
        f"\n[concurrent] {n_req} req | wall {elapsed:.2f}s | "
        f"{n_req / elapsed:.2f} req/s | {total_tokens / elapsed:.1f} tok/s"
    )


# ---------------------------------------------------------------------------
# Mode 1: explicit task:concurrent chat cases over real HTTP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_concurrent_server():
    """One shared HTTP server plus one document directory for all scenarios."""
    if not _CONCURRENT_CASES:
        yield None
        return
    serve = _CFG.serve
    base_args = [*_CFG.serve_args(**serve.extra_engine), *serve.extra_args]
    extra_args = assemble_gate_extra_args(base_args, _CONCURRENT_CASES, _REPO_ROOT)
    document_dir = gate_results_root(
        _REPO_ROOT,
        run_id=_GATE_RUN_ID,
        platform=_PLATFORM,
        device=_DEVICE,
        model=_MODEL,
        case=f"{_CASE}-concurrent",
    )
    ensure_clean_gate_directory(document_dir)
    scenarios = tuple(
        (scenario, len(_SCENARIO_GROUPS[scenario])) for scenario in _SCENARIOS
    )
    with (
        VllmServer(
            model=_CFG.model,
            tp_size=_CFG.engine.get("tensor_parallel_size", 1),
            api_key=serve.api_key,
            served_model_name=serve.served_model_name,
            max_retries=serve.startup_retries,
            extra_args=extra_args,
        ) as server,
        temporary_environment(
            {
                "NO_PROXY": append_loopback_no_proxy(os.environ.get("NO_PROXY")),
                "no_proxy": append_loopback_no_proxy(os.environ.get("no_proxy")),
            }
        ),
    ):
        yield server, document_dir
        with server.log_on_failure("Concurrent gate contract failed"):
            summary = validate_scenario_documents(document_dir, scenarios)
        print(format_gate_total_log("concurrent", summary), flush=True)


def _run_gate_scenario(scenario, gate_concurrent_server) -> None:
    """Eight real HTTP requests for one scenario, gathered once."""
    server, document_dir = gate_concurrent_server
    entries = [gate_entry(case, _REPO_ROOT) for case in _SCENARIO_GROUPS[scenario]]
    with server.log_on_failure(f"Concurrent gate scenario '{scenario}' failed"):
        ensure_gate_server_alive(server, "concurrent")
        outputs = run_concurrent_scenario(
            entries,
            served_model_name=_REQUEST_MODEL,
            base_url=server.base_url,
            api_key=_CFG.serve.api_key,
            timeout=resolve_request_timeout(os.environ),
            document_dir=document_dir,
            scenario=scenario,
            model_path=_CFG.model,
            port=server.port,
        )
    print(format_gate_scenario_log("concurrent", scenario, outputs), flush=True)
    assert all(output["passed"] for output in outputs), (
        f"Concurrent gate scenario '{scenario}' failed; "
        f"details:\n{gate_failure_details(outputs)}"
    )


# ---------------------------------------------------------------------------
# Unified test: mode dispatch by YAML configuration
# ---------------------------------------------------------------------------

if _CONCURRENT_CASES:
    _SCENARIO_PARAMS = list(_SCENARIOS)
    _SCENARIO_IDS = list(_SCENARIOS)
else:
    _SCENARIO_PARAMS = [None]
    _SCENARIO_IDS = ["offline"]


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", _SCENARIO_PARAMS, ids=_SCENARIO_IDS)
def test_concurrent(scenario, gate_concurrent_server) -> None:
    """Dispatch to the HTTP gate path or the classic offline smoke."""
    if _CONCURRENT_CASES:
        _run_gate_scenario(scenario, gate_concurrent_server)
        return
    print(f"\n[{_MODEL}/{_CASE}] concurrent_n: {CONCURRENT_N}")
    print(f"[{_MODEL}/{_CASE}] model: {_CFG.model}")
    _visible_env_var = get_visible_device_env_var()
    print(
        f"[{_MODEL}/{_CASE}] devices: "
        f"{_visible_env_var}={os.environ.get(_visible_env_var, '(not set)')}"
    )

    # Cycle the prompt pool up to N requests.
    items = [_TEXT_CASES[i % len(_TEXT_CASES)] for i in range(CONCURRENT_N)]
    prompts = [case["text"] for case in items]

    llm = None
    try:
        llm = LLM(**_CFG.engine_kwargs())
        sampling_params = SamplingParams(**_SAMPLING)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        assert len(outputs) == len(prompts), (
            f"Expected {len(prompts)} outputs, got {len(outputs)}"
        )

        total_tokens = 0
        for i, (case, output) in enumerate(zip(items, outputs)):
            text = output.outputs[0].text
            total_tokens += len(output.outputs[0].token_ids)

            assert text.strip(), f"Empty output for request[{i}]: {case['text']}"
            print(f"  request[{i}]: {case['text']!r}")
            print(f"  output[{i}]: {text!r}")

            if case["expected"]:
                lower = text.lower()
                matched = any(item.lower() in lower for item in case["expected"])
                assert matched, (
                    f"Expected one of {case['expected']!r} in output for "
                    f"request[{i}], got: {text!r}"
                )

        _report(CONCURRENT_N, elapsed, total_tokens)
    finally:
        if llm is not None:
            # Explicitly shut down the engine core so that the executor calls
            # waitpid() on all worker processes before we return.  Relying on
            # `del llm` alone is not sufficient: Python's GC does not guarantee
            # that __del__ fires immediately, which leaves worker processes as
            # zombies parented by init when the test runner exits.
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
            del llm
        _prepare_cleanup_for_platform()
        cleanup_dist_env_and_memory()
