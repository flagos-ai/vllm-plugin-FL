# Copyright (c) 2025 BAAI. All rights reserved.

"""
Concurrent generation smoke test driven by model YAML configs.

Adapted from the sglang-plugin-FL concurrent smoke test: N requests are
submitted to a single engine in one ``LLM.generate`` call, so the scheduler
runs them through continuous batching concurrently, then every output is
validated. The ``concurrent`` block in the model YAML is optional —
defaults keep the CI smoke small (8 requests, 64 tokens each).

This test file is driven by environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3_6``)
- ``FL_TEST_CASE``: Case name within the family (e.g. ``27b_tp2_eager``)
- ``FL_TEST_PLATFORM``/``FL_TEST_DEVICE``: Apply platform device overrides

Overrides (optional): ``FL_CONCURRENT_N``, ``FL_CONCURRENT_MAX_TOKENS``.
"""

import contextlib
import os
import time

import pytest

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

if not os.path.exists(_CFG.model):
    pytest.fail(
        f"Model not found: {_CFG.model}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Prompt pool and sampling
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


@pytest.mark.e2e
def test_concurrent() -> None:
    """Smoke test: N concurrent requests through one generate call."""
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
