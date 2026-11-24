# Copyright (c) 2026 BAAI. All rights reserved.
"""Coverage for vllm_fl.patches.minicpmo_audio (real import + monkeypatch)."""

import sys
from types import ModuleType

from vllm_fl.patches import minicpmo_audio as mcp

_LEAF = "vllm.model_executor.models.minicpmo"


class _BaseProcessingInfo:
    """Stand-in for the upstream MiniCPM-V image/video-only estimate."""

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": seq_len, "video": mm_counts.get("video", 0)}

    def get_max_audio_tokens(self):
        return 1234


def _install_processing_info(monkeypatch, processing_info):
    """Inject MiniCPMOProcessingInfo under the real dotted import path."""
    parts = _LEAF.split(".")
    parent = None
    for index in range(2, len(parts) + 1):
        name = ".".join(parts[:index])
        module = ModuleType(name)
        module.__path__ = []
        if parent is not None:
            setattr(parent, parts[index - 1], module)
        monkeypatch.setitem(sys.modules, name, module)
        parent = module
    parent.MiniCPMOProcessingInfo = processing_info
    return parent


def test_patch_adds_audio_to_inherited_budget(monkeypatch):
    class Info(_BaseProcessingInfo):
        pass

    _install_processing_info(monkeypatch, Info)
    # No native override yet: the patch must install one.
    assert "get_mm_max_tokens_per_item" not in Info.__dict__

    assert mcp.patch_minicpmo_audio_budget() is True
    assert "get_mm_max_tokens_per_item" in Info.__dict__

    merged = Info().get_mm_max_tokens_per_item(16, {"video": 2})
    assert merged == {"image": 16, "video": 2, "audio": 1234}

    # Repeat registration must be a no-op, not a second wrapping.
    assert mcp.patch_minicpmo_audio_budget() is False
    assert Info().get_mm_max_tokens_per_item(3, {})["audio"] == 1234


def test_patch_preserves_native_override(monkeypatch):
    def native(self, seq_len, mm_counts):
        return {"image": 1}

    class Info(_BaseProcessingInfo):
        get_mm_max_tokens_per_item = native

    _install_processing_info(monkeypatch, Info)
    assert mcp.patch_minicpmo_audio_budget() is False
    assert Info.get_mm_max_tokens_per_item is native
    assert Info().get_mm_max_tokens_per_item(9, {}) == {"image": 1}
