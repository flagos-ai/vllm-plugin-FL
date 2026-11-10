"""Exercise the upstream selection property, without assigning its result."""

import pytest

from vllm.config import VllmConfig, vllm as config_module


@pytest.mark.parametrize("explicit,expected", [(None, True), ("0", False), ("1", True)])
def test_upstream_runner_selection(monkeypatch, explicit, expected):
    if explicit is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", explicit)
    monkeypatch.setattr(config_module, "HAS_TRITON", True)
    config = VllmConfig()
    assert config.use_v2_model_runner is expected


def test_upstream_fallback_when_triton_unavailable(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(config_module, "HAS_TRITON", False)
    assert VllmConfig().use_v2_model_runner is False


def test_upstream_unsupported_feature_fallback(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(config_module, "HAS_TRITON", True)
    config = VllmConfig()
    config.parallel_config.distributed_executor_backend = "external_launcher"
    config.parallel_config.pipeline_parallel_size = 2
    assert "pipeline parallelism with external_launcher" in (
        config._get_v2_model_runner_unsupported_features()
    )
    assert config.use_v2_model_runner is False
