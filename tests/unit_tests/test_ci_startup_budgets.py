# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU regressions for platform-scoped cold-start budgets."""

import importlib.util
from pathlib import Path

import pytest

from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.model_config import ModelConfig


@pytest.mark.parametrize("inherit_environment", [False, True])
@pytest.mark.parametrize(
    "case",
    ["27b_tp2_eager", "27b_tp2_graph", "35b_a3b_tp2_eager", "35b_a3b_tp2_graph"],
)
def test_s60_startup_budget_is_scoped(case, inherit_environment, monkeypatch):
    monkeypatch.delenv("FL_TEST_PLATFORM", raising=False)
    monkeypatch.delenv("FL_TEST_DEVICE", raising=False)
    baseline = ModelConfig.load("qwen3_6", case)
    if inherit_environment:
        monkeypatch.setenv("FL_TEST_PLATFORM", "enflame")
        monkeypatch.setenv("FL_TEST_DEVICE", "s60")
        config = ModelConfig.load("qwen3_6", case)
    else:
        config = ModelConfig.load("qwen3_6", case, platform="enflame", device="s60")
    assert baseline.serve.startup_retries == 120
    assert config.serve.startup_retries == 180
    server = VllmServer(model=config.model, max_retries=config.serve.startup_retries)
    assert server.max_retries * server.poll_interval == 30 * 60
    # Readiness is a harness option, never an engine/CLI option.
    assert "--startup-retries" not in config.serve_args()
    assert config.serve.chat_cases == baseline.serve.chat_cases
    assert config.serve.max_tokens == baseline.serve.max_tokens


def test_thead_inference_job_budget_reaches_matrix():
    path = Path(__file__).parents[2] / ".github/scripts/generate_matrix.py"
    spec = importlib.util.spec_from_file_location("_budget_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_platform("thead")
    assert module.resolve_e2e_timeout(config, "zw810e", "inference") == 180
    assert module.resolve_e2e_timeout(config, "zw810e", "serving") == 180
    assert module.resolve_e2e_timeout({}, "zw810e", "inference") == 60
