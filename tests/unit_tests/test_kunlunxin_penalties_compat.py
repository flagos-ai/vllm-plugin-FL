# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Test the Kunlunxin V2 penalty provider bindings."""

import sys
from types import ModuleType
from unittest.mock import Mock

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import penalties_compat


def test_apply_penalties_compat_binds_provider(monkeypatch):
    module = ModuleType("vllm.v1.worker.gpu.sample.penalties")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    provider = ModuleType("flaggems_vllm")
    provider.v2_apply_penalties = Mock()
    provider.v2_penalty_bincount = Mock()
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    penalties_compat.apply_penalties_compat()
    assert module.apply_penalties is provider.v2_apply_penalties
    assert module.bincount is provider.v2_penalty_bincount

    original_apply = provider.v2_apply_penalties
    original_count = provider.v2_penalty_bincount
    provider.v2_apply_penalties = Mock()
    provider.v2_penalty_bincount = Mock()
    penalties_compat.apply_penalties_compat()
    assert module.apply_penalties is original_apply
    assert module.bincount is original_count


def test_apply_penalties_compat_tolerates_an_absent_v2_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.penalties", None)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    penalties_compat.apply_penalties_compat()
