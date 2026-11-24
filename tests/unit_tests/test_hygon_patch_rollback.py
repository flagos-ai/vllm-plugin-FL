# Copyright (c) 2026 BAAI. All rights reserved.

"""Failure semantics of the Hygon startup patches.

The patches are applied from ``register()`` before model construction, and
vLLM's platform probing catches registration errors and keeps going. A patch
set that is half applied and half rolled back therefore has to be impossible:
either every change lands, or the process is restored exactly as it was and a
later call retries from a clean state.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def hygon_patch():
    # Load only the patch source: importing vllm_fl initializes every dispatch
    # backend, which this compatibility contract does not need.
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/hygon/patch.py"
    )
    spec = importlib.util.spec_from_file_location("hygon_patch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_hygon_patches.cache_clear()
    yield module
    module.apply_hygon_patches.cache_clear()


@pytest.fixture
def vendor_modules(monkeypatch):
    """Fake torch and vLLM modules, isolated from the real ones."""

    class RejectUvaBinding(ModuleType):
        """Module whose ``is_uva_available`` binding cannot be replaced."""

        def __setattr__(self, name, value):
            if name == "is_uva_available":
                raise RuntimeError("read-only binding")
            super().__setattr__(name, value)

    for name in [n for n in list(sys.modules) if n == "vllm" or n.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)

    accelerator = SimpleNamespace()
    torch = ModuleType("torch")
    torch.accelerator = accelerator
    torch.cuda = SimpleNamespace(mem_get_info=lambda index=None: (4096, 8192))
    monkeypatch.setitem(sys.modules, "torch", torch)

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    platform_utils = ModuleType("vllm.utils.platform_utils")
    platform_utils.is_uva_available = lambda: True
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.platform_utils", platform_utils)

    return SimpleNamespace(
        torch=torch,
        accelerator=accelerator,
        platform_utils=platform_utils,
        reject_binding=RejectUvaBinding,
    )


def test_apply_installs_every_patch_and_marks_it(hygon_patch, vendor_modules):
    hygon_patch.apply_hygon_patches()

    assert vendor_modules.accelerator.get_memory_info() == (4096, 8192)
    assert vendor_modules.accelerator._hcu_attrs_patched is True
    assert vendor_modules.platform_utils.is_uva_available() is False
    assert vendor_modules.platform_utils._hcu_uva_patched is True


def test_accelerator_change_is_rolled_back_when_the_uva_patch_fails(
    hygon_patch, vendor_modules, monkeypatch
):
    def fail():
        raise RuntimeError("UVA patch failed")

    monkeypatch.setattr(hygon_patch, "_install_uva_unavailable", fail)

    with pytest.raises(RuntimeError, match="UVA unavailable"):
        hygon_patch.apply_hygon_patches()

    assert "get_memory_info" not in vars(vendor_modules.accelerator)
    assert not hasattr(vendor_modules.accelerator, "_hcu_attrs_patched")


def test_uva_rebinding_is_rolled_back_when_a_module_rejects_it(
    hygon_patch, vendor_modules, monkeypatch
):
    original = vendor_modules.platform_utils.is_uva_available
    hostile = vendor_modules.reject_binding("vllm.hostile")
    ModuleType.__setattr__(hostile, "is_uva_available", original)
    monkeypatch.setitem(sys.modules, "vllm.hostile", hostile)

    with pytest.raises(RuntimeError, match="UVA unavailable"):
        hygon_patch.apply_hygon_patches()

    assert vendor_modules.platform_utils.is_uva_available is original
    assert not hasattr(vendor_modules.platform_utils, "_hcu_uva_patched")
    assert "get_memory_info" not in vars(vendor_modules.accelerator)

    # With the faulty module gone, the same entry point retries from scratch.
    monkeypatch.delitem(sys.modules, "vllm.hostile")
    hygon_patch.apply_hygon_patches()

    assert vendor_modules.accelerator.get_memory_info() == (4096, 8192)
    assert vendor_modules.platform_utils.is_uva_available() is False
    assert vendor_modules.platform_utils._hcu_uva_patched is True


def test_existing_memory_info_is_never_shadowed(hygon_patch, vendor_modules):
    vendor_impl = lambda index=None: (1024, 2048)  # noqa: E731
    vendor_modules.accelerator.get_memory_info = vendor_impl

    hygon_patch.apply_hygon_patches()

    assert vendor_modules.accelerator.get_memory_info is vendor_impl
    assert vendor_modules.accelerator._hcu_attrs_patched is True


def test_repeated_application_is_idempotent(hygon_patch, vendor_modules):
    hygon_patch.apply_hygon_patches()
    installed = vars(vendor_modules.accelerator).copy()

    hygon_patch.apply_hygon_patches()

    assert vars(vendor_modules.accelerator) == installed
