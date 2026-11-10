# Copyright (c) 2026 BAAI. All rights reserved.

"""The FlagGems backend must survive on images that ship ``flag_gems`` but not
the optional ``flaggems_vllm`` provider.

Before this, a missing provider raised ``ImportError`` inside
``register_builtins`` and the whole FlagGems backend was dropped, so
``attention_backend`` silently fell back to the reference FlASH_ATTN path.
"""

import contextlib
import importlib
import sys

import pytest

pytest.importorskip("vllm")

_PROVIDER = "flaggems_vllm"
_PKG = "vllm_fl.dispatch.backends.flaggems"


class _BlockProvider:
    """A meta-path finder that makes ``import flaggems_vllm`` fail."""

    def find_spec(self, name, path, target=None):
        if name == _PROVIDER or name.startswith(_PROVIDER + "."):
            raise ImportError(f"{name} blocked for test")
        return None


def _purge(prefix: str) -> None:
    for mod in list(sys.modules):
        if mod == prefix or mod.startswith(prefix + "."):
            del sys.modules[mod]


@pytest.fixture
def without_provider():
    """Run the body with ``flaggems_vllm`` unimportable and the FlagGems
    backend modules freshly re-imported under that condition."""
    blocker = _BlockProvider()
    sys.meta_path.insert(0, blocker)
    _purge(_PROVIDER)
    _purge(_PKG)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(blocker)
        # Let subsequent tests re-import against the real provider.
        _purge(_PKG)


def test_impl_modules_import_without_provider(without_provider):
    # The attention/mla impl modules are pulled in transitively by
    # impl/__init__, so a module-level provider import would abort the backend.
    attn = importlib.import_module(f"{_PKG}.impl.attention")
    mla = importlib.import_module(f"{_PKG}.impl.mla")
    with pytest.raises(ImportError):
        attn.flash_attn_varlen_func()
    with pytest.raises(ImportError):
        mla.flash_mla()


def test_register_keeps_backend_without_provider(without_provider):
    from vllm_fl.dispatch.registry import OpRegistry

    register_ops = importlib.import_module(f"{_PKG}.register_ops")
    assert register_ops._provider_installed() is False

    registry = OpRegistry()
    register_ops.register_builtins(registry)  # must not raise

    ops = set(registry.list_operators())
    assert {"rms_norm", "attention_backend", "silu_and_mul"} <= ops

    # A provider-backed op stays registered but reports unavailable, so dispatch
    # falls through to the next backend instead of the whole backend vanishing.
    silu = registry.get_implementation("silu_and_mul", "default.flagos")
    assert silu.is_available() is False

    # rms_norm and attention_backend need only flag_gems, so their availability
    # tracks flag_gems and is independent of the missing provider.
    try:
        import flag_gems  # noqa: F401

        has_flag_gems = True
    except Exception:
        has_flag_gems = False
    if has_flag_gems:
        rms = registry.get_implementation("rms_norm", "default.flagos")
        assert rms.is_available() is True
