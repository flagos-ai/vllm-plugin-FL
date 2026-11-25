# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType

from vllm_fl.dispatch.backends.flaggems.impl import mla_sparse


def test_sparse_mla_prefers_flaggems_vllm(monkeypatch):
    package = ModuleType("flaggems_vllm")

    def specialized_impl(**kwargs):
        return kwargs

    package.flash_mla_sparse_fwd = specialized_impl
    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)

    assert mla_sparse._resolve_flash_mla_sparse_fwd() is specialized_impl


def test_sparse_mla_falls_back_when_specialized_symbol_is_missing(monkeypatch):
    specialized_package = ModuleType("flaggems_vllm")
    generic_package = ModuleType("flag_gems")

    def generic_impl(**kwargs):
        return kwargs

    generic_package.flash_mla_sparse_fwd = generic_impl
    monkeypatch.setitem(sys.modules, "flaggems_vllm", specialized_package)
    monkeypatch.setitem(sys.modules, "flag_gems", generic_package)

    assert mla_sparse._resolve_flash_mla_sparse_fwd() is generic_impl
