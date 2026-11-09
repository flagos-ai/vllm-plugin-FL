"""Worker lifecycle regressions using real tensors and vLLM 0.32 configs.

CPU tests exercise worker methods without allocating a device/model; they do
not replace vendor model-generation acceptance.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import LoadConfig
from vllm_fl.worker.worker import WorkerFL


def test_update_config_propagates_same_object():
    config = LoadConfig()
    worker = object.__new__(WorkerFL)
    worker.vllm_config = SimpleNamespace(load_config=config)
    worker.load_config = config
    worker.model_runner = SimpleNamespace(load_config=config)
    worker.update_config({"load_config": {"download_dir": "/tmp/new-model-cache"}})
    assert worker.load_config.download_dir == "/tmp/new-model-cache"
    assert worker.load_config is worker.vllm_config.load_config
    assert worker.load_config is worker.model_runner.load_config
    with pytest.raises(ValueError, match="not supported"):
        worker.update_config({"cache_config": {}})


@pytest.mark.parametrize("tags", [None, ["kv_cache"], ["weights"], []])
@pytest.mark.parametrize("dtype", ["fp8", "auto"])
def test_wake_restores_buffers_and_initializes_only_fp8_kv(monkeypatch, tags, dtype):
    worker = object.__new__(WorkerFL)
    scale = torch.tensor([0.375])
    model = torch.nn.Module()
    model.register_buffer("scale", scale)
    kv = torch.full((4,), 7.0)
    hybrid = torch.full((2,), 8.0)
    worker.model_runner = SimpleNamespace(model=model, kv_caches=[kv, None, [hybrid]])
    worker.cache_config = SimpleNamespace(cache_dtype=dtype)
    worker._sleep_saved_buffers = {"scale": torch.tensor([0.125])}
    worker._sleep_saved_draft_buffers = {}
    worker._sleep_saved_kv_scales = [(scale, torch.tensor([0.125]))]
    backend = Mock()
    worker._sleep_mode_backend = backend
    sync = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    worker.wake_up(tags)
    backend.resume.assert_called_once_with(tags)
    sync.assert_called_once()
    weights = tags is None or "weights" in tags
    assert scale.item() == (0.125 if weights else 0.375)
    cleared = dtype == "fp8" and (tags is None or "kv_cache" in tags)
    assert torch.all(kv == (0 if cleared else 7))
    assert torch.all(hybrid == (0 if cleared else 8))


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_wake_preserves_calibrated_scale_alias(monkeypatch, level):
    worker = object.__new__(WorkerFL)
    scale = torch.tensor([0.0625])
    model = torch.nn.Module()
    model.register_buffer("scale", scale)
    kv = torch.full((2,), 9.0)
    worker.model_runner = SimpleNamespace(model=model, kv_caches=[kv])
    worker.cache_config = SimpleNamespace(cache_dtype="fp8")
    worker.compilation_config = SimpleNamespace(
        static_forward_context={"attn": SimpleNamespace(_k_scale=scale, k_scale=scale)}
    )
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.get_draft_model = lambda: None
    backend = Mock()
    backend.suspend.side_effect = lambda level: scale.fill_(99)
    worker._sleep_mode_backend = backend
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (100, 200))
    worker.sleep(level)
    assert len(worker._sleep_saved_kv_scales) == 1
    worker.wake_up(["kv_cache"])
    assert scale.item() == 99
    assert torch.all(kv == 0)
    worker.wake_up(["weights"])
    assert scale.item() == 0.0625
    assert worker._sleep_saved_kv_scales == []
