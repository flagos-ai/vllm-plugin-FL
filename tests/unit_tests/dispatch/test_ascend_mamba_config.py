# Copyright (c) 2026 BAAI. All rights reserved.

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("torch_npu")

from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

_UPSTREAM_VERIFIER = HybridAttentionMambaModelConfig.verify_and_update_config.__func__

import vllm_fl.dispatch.backends.vendor.ascend as ascend_backend
from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size


def _make_vllm_config():
    cache_config = SimpleNamespace(
        block_size=1152,
        mamba_page_size_padded=4726784,
        mamba_block_size=1152,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe"),
        is_hybrid=True,
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )


def test_ascend_import_keeps_upstream_hybrid_config_verifier():
    assert (
        HybridAttentionMambaModelConfig.verify_and_update_config.__func__
        is _UPSTREAM_VERIFIER
    )

    importlib.reload(ascend_backend)

    assert (
        HybridAttentionMambaModelConfig.verify_and_update_config.__func__
        is _UPSTREAM_VERIFIER
    )


def test_refresh_block_size_preserves_upstream_hybrid_alignment():
    vllm_config = _make_vllm_config()

    refresh_block_size(vllm_config)

    assert vllm_config.cache_config.block_size == 1152
    assert vllm_config.cache_config.mamba_block_size == 1152
    assert vllm_config.cache_config.mamba_page_size_padded == 4726784
