# Copyright (c) 2026 BAAI. All rights reserved.
"""Extra coverage for base.py and flaggems.py delegation branches."""

import pytest
import torch

from vllm_fl.dispatch.backends.base import Backend
from vllm_fl.dispatch.backends.flaggems import flaggems as fg
from vllm_fl.dispatch.backends.flaggems.impl import (
    fused_moe as fg_fused_moe,
    rotary as fg_rotary,
)
from vllm_fl.dispatch.backends.reference import reference as ref


def test_backend_is_abstract_and_vendor_defaults_to_none():
    class Missing(Backend):
        pass

    with pytest.raises(TypeError):
        Missing()

    class Complete(Backend):
        def is_available(self) -> bool:
            return True

        @property
        def name(self) -> str:
            return "complete"

    assert Complete().vendor is None
    assert ref.ReferenceBackend().vendor is None
    assert fg.FlagGemsBackend().vendor is None


def test_flaggems_availability_caches_successful_import(monkeypatch):
    monkeypatch.setattr(fg.FlagGemsBackend, "_available", None)
    backend = fg.FlagGemsBackend()
    assert backend.is_available() is True
    assert fg.FlagGemsBackend._available is True
    assert backend.is_available() is True


def test_moe_align_forwards_every_argument(monkeypatch):
    seen = []

    def recorder(*args):
        seen.append(args)
        return "aligned"

    monkeypatch.setattr(
        fg_fused_moe, "moe_align_block_size_flaggems", recorder
    )
    topk_ids = torch.zeros((1, 2), dtype=torch.int32)
    expert_map = torch.arange(2, dtype=torch.int32)
    result = fg.FlagGemsBackend().moe_align_block_size(
        topk_ids,
        4,
        2,
        expert_map=expert_map,
        pad_sorted_ids=True,
        ignore_invalid_experts=True,
    )
    assert result == "aligned"
    args = seen[0]
    assert args[0] is topk_ids
    assert args[1] == 4 and args[2] == 2
    assert args[3] is expert_map
    assert args[4] is True and args[5] is True


def test_rotary_forwards_flags(monkeypatch):
    seen = {}

    def recorder(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "rotated"

    monkeypatch.setattr(fg_rotary, "rotary_embedding_flaggems", recorder)
    q = torch.zeros((1, 1, 4))
    position_ids = torch.zeros(1, dtype=torch.long)
    result = fg.FlagGemsBackend().rotary_embedding(
        None,
        q,
        q,
        q,
        q,
        position_ids,
        rotary_interleaved=True,
        inplace=False,
    )
    assert result == "rotated"
    assert seen["kwargs"]["rotary_interleaved"] is True
    assert seen["kwargs"]["inplace"] is False


def test_attention_backend_env_one_is_truthy(monkeypatch):
    monkeypatch.setattr(fg.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    backend = fg.FlagGemsBackend().attention_backend()
    assert backend.endswith("attention.AttentionFLBackend")
