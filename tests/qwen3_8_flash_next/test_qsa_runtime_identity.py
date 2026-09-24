# SPDX-License-Identifier: Apache-2.0
"""Runtime identity follows the callable used by the local indexer."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_fl.models.qwen3_8_flash_next.gpu import indexer_qsa
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa
from vllm_fl.models.qwen3_8_flash_next.vendor.vllm024.dispatch import (
    qwen4_qsa_pre_indexer_status,
)


@pytest.mark.parametrize(
    "enabled,dim,neox,fused",
    [
        (True, 128, True, True),
        (False, 128, True, False),
        (True, 64, True, False),
        (True, 128, False, False),
    ],
)
def test_compression_status_uses_execution_selector(
    monkeypatch, enabled, dim, neox, fused
):
    monkeypatch.setattr(indexer_qsa, "_QSA_FUSED_COMPRESS_ENABLED", enabled)
    instance = indexer_qsa.QSAIndexer.__new__(indexer_qsa.QSAIndexer)
    instance.index_head_dim = dim
    instance.select_all_tokens = False
    instance.rotary_emb = SimpleNamespace(rotary_dim=64, is_neox_style=neox)
    selected = instance._compression_impl()
    assert selected is (
        qsa.qsa_compress_norm_mrope_store_groups
        if fused
        else qsa.qsa_compress_groups_with_ratio
    )
    status = instance.runtime_status()
    identity = status["stages"]["compression"]
    assert identity["callable"] == f"{selected.__module__}.{selected.__qualname__}"
    assert (
        identity["source_sha256"]
        == hashlib.sha256(Path(qsa.__file__).read_bytes()).hexdigest()
    )
    assert status["compression_candidates"] == []


def test_reference_only_vendor_is_never_reported_as_enabled():
    status = qwen4_qsa_pre_indexer_status()
    assert status["enabled"] is False
    assert status["backend"] == "local_qsa_composition"
    assert status["runtime"]["official_pre_indexer_enabled"] is False
    assert len(status["runtime"]["compression_candidates"]) == 2
    for identity in status["runtime"]["stages"].values():
        assert ".gpu.ops.qsa." in identity["callable"]
        assert identity["source_sha256"]
