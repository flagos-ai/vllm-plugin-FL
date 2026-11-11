# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin UVA probe override.

P800 exposes pinned host memory but cannot map it into the device address
space, so vLLM's ``is_uva_available()`` (defined as ``is_pin_memory_available()
or current_platform.is_cpu()``) must be forced to False for the V2 model runner
to pick its explicit-copy buffers instead of ``UvaBuffer``.
"""

import sys
from functools import cache
from types import ModuleType

import pytest
from vllm.utils import platform_utils

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.uva_compat import (
    apply_uva_compat,
)

_IMPORTER = "vllm.v1.worker.gpu.buffer_utils"


@pytest.fixture
def cached_probe(monkeypatch):
    """Install a fresh functools.cache-wrapped probe that reports True."""

    @cache
    def is_uva_available() -> bool:
        return True

    monkeypatch.setattr(platform_utils, "is_uva_available", is_uva_available)
    return is_uva_available


def test_probe_reports_false_after_patch(cached_probe):
    assert platform_utils.is_uva_available() is True

    apply_uva_compat()

    assert platform_utils.is_uva_available() is False


def test_memoized_true_is_discarded(cached_probe):
    # Something queried UVA before the patch ran, so functools.cache already
    # holds True; the override must invalidate it rather than inherit it.
    cached_probe()
    assert cached_probe.cache_info().currsize == 1

    apply_uva_compat()

    assert cached_probe.cache_info().currsize == 0
    assert platform_utils.is_uva_available() is False


def test_rebinds_already_imported_by_value_importers(monkeypatch, cached_probe):
    # buffer_utils does `from vllm.utils.platform_utils import is_uva_available`,
    # so patching only the defining module would leave it calling the original.
    stub = ModuleType(_IMPORTER)
    stub.is_uva_available = cached_probe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _IMPORTER, stub)

    apply_uva_compat()

    assert stub.is_uva_available() is False
    assert stub.is_uva_available is platform_utils.is_uva_available


def test_skips_importers_that_are_not_loaded(monkeypatch, cached_probe):
    # Importing them here would drag triton and the GPU worker stack in far
    # earlier than vLLM intends, so absent modules are left alone.
    monkeypatch.delitem(sys.modules, _IMPORTER, raising=False)

    apply_uva_compat()

    assert _IMPORTER not in sys.modules
    assert platform_utils.is_uva_available() is False


def test_does_not_override_a_foreign_rebind(monkeypatch, cached_probe):
    # A module that already points somewhere else is not ours to redirect.
    def other_probe() -> bool:
        return True

    stub = ModuleType(_IMPORTER)
    stub.is_uva_available = other_probe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _IMPORTER, stub)

    apply_uva_compat()

    assert stub.is_uva_available is other_probe


def test_patch_is_idempotent(cached_probe):
    apply_uva_compat()
    patched = platform_utils.is_uva_available

    apply_uva_compat()

    assert platform_utils.is_uva_available is patched
    assert patched.__wrapped_original__ is cached_probe


def test_pin_memory_probe_is_left_alone(cached_probe):
    # Pinned staging buffers and async H2D copies do work on P800; only the
    # mapping is unsupported.
    before = platform_utils.is_pin_memory_available

    apply_uva_compat()

    assert platform_utils.is_pin_memory_available is before
