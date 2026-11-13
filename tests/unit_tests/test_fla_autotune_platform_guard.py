# Copyright (c) 2026 BAAI. All rights reserved.

"""The GCU autotuning hook must be harmless on other hardware backends."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("gcu_state", ["missing", "unavailable"])
def test_fla_autotune_without_gcu(monkeypatch, gcu_state):
    # Load the actual module without importing vendor package initializers.
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/gcu/impl/fla_autotune.py"
    )
    spec = importlib.util.spec_from_file_location("_fla_autotune_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if gcu_state == "missing":
        monkeypatch.delattr(torch, "gcu", raising=False)
    else:
        monkeypatch.setattr(
            torch, "gcu", SimpleNamespace(is_available=lambda: False), raising=False
        )
    module.apply_fla_autotune_gcu_patch()
    assert module._patched is False
