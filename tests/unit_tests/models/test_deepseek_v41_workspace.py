# SPDX-License-Identifier: Apache-2.0
"""The configured indexer and request admission must use the same logical limit."""

from types import SimpleNamespace

import pytest

from vllm_fl.models.deepseek_v41.config import INDEXER_LIMIT, indexer_workspace_tokens


def config(limit=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=1048576, hf_config=SimpleNamespace(model_type="deepseek_v41")
        ),
        cache_config=SimpleNamespace(block_size=64),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        additional_config={} if limit is None else {INDEXER_LIMIT: limit},
    )


@pytest.mark.parametrize(
    "limit,expected", [(None, 1048576), (66560, 66624), (1048576, 1048576)]
)
def test_workspace_keeps_model_context_and_speculative_tail(limit, expected):
    cfg = config(limit)
    assert indexer_workspace_tokens(cfg) == expected
    assert cfg.model_config.max_model_len == 1048576


@pytest.mark.parametrize("limit", [True, 32704, 66561, 1048640])
def test_invalid_workspace(limit):
    with pytest.raises(ValueError, match=INDEXER_LIMIT):
        indexer_workspace_tokens(config(limit))


def test_admission_preserves_native_request_and_wave(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    from vllm_fl.dispatch.backends.vendor.metax.patches.deepseek_v41 import indexer

    def preprocess(self, request):
        return request, 3

    monkeypatch.setattr(EngineCore, "preprocess_add_request", preprocess)
    indexer.install_indexer_request_guard()
    bound = EngineCore.preprocess_add_request
    indexer.install_indexer_request_guard()
    assert EngineCore.preprocess_add_request is bound
    engine = SimpleNamespace(vllm_config=config(66560))
    request = SimpleNamespace(
        sampling_params=object(),
        num_prompt_tokens=65536,
        max_tokens=1024,
        resumable=False,
    )
    assert bound(engine, request) == (request, 3)
    request.max_tokens += 1
    with pytest.raises(ValueError, match="66560"):
        bound(engine, request)
    request.max_tokens = 1
    request.resumable = True
    with pytest.raises(ValueError, match="complete input"):
        bound(engine, request)
    assert bound(SimpleNamespace(vllm_config=config()), request) == (request, 3)


@pytest.mark.parametrize(
    "live_after,free_after,expected", [(100, 35, 15), (99, 35, None), (100, 60, None)]
)
def test_graph_accounting_preserves_live_allocations(
    monkeypatch, live_after, free_after, expected
):
    from vllm_fl.worker import deepseek_v41 as runtime

    frees, lives = iter([20, free_after]), iter([100, live_after])
    monkeypatch.setattr(
        runtime.GPUModelRunner, "capture_model", lambda self, *, profile_only: 30
    )
    monkeypatch.setattr(runtime.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        runtime.torch.accelerator, "get_memory_info", lambda: (next(frees), 120)
    )
    monkeypatch.setattr(runtime.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(runtime.torch.cuda, "memory_allocated", lambda: next(lives))
    runner = object.__new__(runtime.DeepseekV41ModelRunner)
    if expected is None:
        with pytest.raises(RuntimeError, match="accounting"):
            runner.capture_model(profile_only=True)
    else:
        assert runner.capture_model(profile_only=True) == expected
