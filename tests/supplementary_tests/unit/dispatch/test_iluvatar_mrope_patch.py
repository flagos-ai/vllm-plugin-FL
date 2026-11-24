# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar M-RoPE patch.

Covers ``vllm_fl/dispatch/backends/vendor/iluvatar/mrope.py``:

* ``_mrope_implementation`` env parsing: default "opaque-v1", "native"
  accepted, anything else raises ValueError. Environment and the
  ``vllm.envs.environment_variables`` dict are saved and restored.
* ``patch_mrope``: non-iluvatar vendor is a no-op (forward_native
  untouched, no env entry added); on the iluvatar path the default
  opaque-v1 mode swaps MRotaryEmbedding.forward_native for the wrapper and
  adds the env-var entry bound to ``_mrope_implementation``; native mode
  unwraps back to ``__wrapped__``; repeated calls are safe.

CPU-only. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

_ENV_VAR = "VLLM_FL_ILUVATAR_MROPE_IMPL"


@pytest.fixture(autouse=True)
def _clean_mrope_env(monkeypatch):
    """Isolate the env var and the vllm.envs dict entry; restore after."""
    import vllm.envs

    monkeypatch.delenv(_ENV_VAR, raising=False)
    envs_saved = dict(vllm.envs.environment_variables)
    envs_had_key = _ENV_VAR in envs_saved
    yield
    # restore the dict exactly: remove what we may have added, put back
    # anything the test replaced
    vllm.envs.environment_variables.clear()
    vllm.envs.environment_variables.update(envs_saved)
    if not envs_had_key:
        vllm.envs.environment_variables.pop(_ENV_VAR, None)
    monkeypatch.delenv(_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _save_and_restore_forward_native():
    """Save MRotaryEmbedding.forward_native; restore after each test."""
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        MRotaryEmbedding,
    )

    original = MRotaryEmbedding.forward_native
    yield original
    MRotaryEmbedding.forward_native = original


class TestMropeImplementation:
    def test_default_is_opaque_v1(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import (
            _mrope_implementation,
        )

        assert os.environ.get(_ENV_VAR) is None
        assert _mrope_implementation() == "opaque-v1"

    def test_native_accepted(self, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import (
            _mrope_implementation,
        )

        monkeypatch.setenv(_ENV_VAR, "native")
        assert _mrope_implementation() == "native"

    @pytest.mark.parametrize(
        "bad", ["", "Opaque-V1", "NATIVE", "reference", "opaque", "v2"]
    )
    def test_invalid_raises_value_error(self, monkeypatch, bad):
        from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import (
            _mrope_implementation,
        )

        monkeypatch.setenv(_ENV_VAR, bad)
        with pytest.raises(ValueError, match="must be 'opaque-v1' or 'native'"):
            _mrope_implementation()


class TestPatchMrope:
    def _patched_platform(self, monkeypatch, vendor_name):
        import vllm.platforms

        # patch_mrope does `from vllm.platforms import current_platform`
        # inside the function body, so patch the module attribute there.
        class _Platform:
            pass

        _Platform.vendor_name = vendor_name
        monkeypatch.setattr(vllm.platforms, "current_platform", _Platform())

    def test_non_iluvatar_vendor_is_noop(self, monkeypatch, _save_and_restore_forward_native):
        import vllm.envs

        import vllm_fl.dispatch.backends.vendor.iluvatar.mrope as mrope_mod
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )

        original = MRotaryEmbedding.forward_native
        self._patched_platform(monkeypatch, "nvidia")
        mrope_mod.patch_mrope()
        assert MRotaryEmbedding.forward_native is original
        assert _ENV_VAR not in vllm.envs.environment_variables

    def test_iluvatar_swaps_forward_native_and_sets_env_entry(
        self, monkeypatch, _save_and_restore_forward_native
    ):
        import vllm.envs

        import vllm_fl.dispatch.backends.vendor.iluvatar.mrope as mrope_mod
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )

        original = MRotaryEmbedding.forward_native
        self._patched_platform(monkeypatch, "iluvatar")
        mrope_mod.patch_mrope()

        patched = MRotaryEmbedding.forward_native
        assert patched is not original
        assert getattr(patched, "_iluvatar_mrope_patched", False)
        assert patched.__wrapped__ is original

        entry = vllm.envs.environment_variables.get(_ENV_VAR)
        assert entry is mrope_mod._mrope_implementation

    def test_iluvatar_repeated_patch_keeps_single_wrapper(
        self, monkeypatch, _save_and_restore_forward_native
    ):
        import vllm_fl.dispatch.backends.vendor.iluvatar.mrope as mrope_mod
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )

        self._patched_platform(monkeypatch, "iluvatar")
        mrope_mod.patch_mrope()
        first = MRotaryEmbedding.forward_native
        mrope_mod.patch_mrope()
        mrope_mod.patch_mrope()
        assert MRotaryEmbedding.forward_native is first
        # single wrapper, not a stack
        assert MRotaryEmbedding.forward_native.__wrapped__ is first.__wrapped__

    def test_native_mode_unwraps_to_original(
        self, monkeypatch, _save_and_restore_forward_native
    ):
        import vllm_fl.dispatch.backends.vendor.iluvatar.mrope as mrope_mod
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )

        original = MRotaryEmbedding.forward_native
        self._patched_platform(monkeypatch, "iluvatar")

        monkeypatch.setenv(_ENV_VAR, "native")
        mrope_mod.patch_mrope()
        assert MRotaryEmbedding.forward_native is original

        # switching back to opaque re-wraps
        monkeypatch.setenv(_ENV_VAR, "opaque-v1")
        mrope_mod.patch_mrope()
        assert MRotaryEmbedding.forward_native is not original
        assert getattr(
            MRotaryEmbedding.forward_native, "_iluvatar_mrope_patched", False
        )

    def test_native_mode_after_opaque_unwraps(
        self, monkeypatch, _save_and_restore_forward_native
    ):
        import vllm_fl.dispatch.backends.vendor.iluvatar.mrope as mrope_mod
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding,
        )

        original = MRotaryEmbedding.forward_native
        self._patched_platform(monkeypatch, "iluvatar")

        mrope_mod.patch_mrope()  # opaque default
        assert MRotaryEmbedding.forward_native is not original

        monkeypatch.setenv(_ENV_VAR, "native")
        mrope_mod.patch_mrope()  # must unwrap via __wrapped__
        assert MRotaryEmbedding.forward_native is original
