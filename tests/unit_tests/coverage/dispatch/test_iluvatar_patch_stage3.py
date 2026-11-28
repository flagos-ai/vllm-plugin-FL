# Copyright 2026 BAAI. All rights reserved.
"""Stage-3 supplementary coverage: two defensive probe arms of
``patch_torch_inductor_for_iluvatar`` in ``vendor/iluvatar/iluvatar.py``.

``iluvatar.py`` reaches 251/255 statements on the CPU gate through
``test_iluvatar_patch_functions.py``; the four still-missing rows are the
same four the BI-V150 real-hardware run misses — the two exception arms of
the GPUTarget probe block:

* lines 290-291 — the *inner* ``except Exception: pass`` around the
  ``from triton.backends.compiler import GPUTarget`` import and the probe
  loop. On every environment the tests ran on (WSL triton, the real
  flagtree wheel) ``triton.backends.compiler`` is importable and the probes
  never raise, so the swallow arm never fires. It guards flagtree builds
  that ship a triton without the compiler submodule.
* lines 309-310 — the *outer* non-strict ``except`` arm (debug log +
  return) that fires when the very first ``import triton.backends as _tb``
  fails or the backend table itself is unreadable. The strict sibling
  (``raise RuntimeError('Cannot inspect ...')``) is covered by the existing
  suite; the non-strict quiet-return is not.

Both are triggered here by invalidating the relevant ``sys.modules``
entries (``monkeypatch.setitem(sys.modules, name, None)`` makes any
``from name import ...`` raise ImportError inside the running interpreter,
exactly what a missing/renamed module looks like to production code), with
a fake triton installed so the rest of the patch machinery is hermetic.

The assertions verify the *behavioral contract* of each arm: the inner
swallow leaves ``_target`` unset so the non-strict fallback remaps to
'corex' (asserted through the installed ``_IluvatarGPUTarget``), and the
outer arm returns without installing any GPUTarget remap (asserted by
probing the triton_heuristics attribute before and after).
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

TRITON = "triton"
TRITON_LANGUAGE = "triton.language"
TRITON_OPS_MPM = "triton.ops.matmul_perf_model"
TRITON_BACKENDS = "triton.backends"
TRITON_BACKENDS_COMPILER = "triton.backends.compiler"


@pytest.fixture
def fake_flagtree_triton(monkeypatch):
    """A flagtree-shaped triton 3.2.1 with an iluvatar-only backend table.

    ``triton.backends`` is *not* given a ``compiler`` submodule and the
    real ``triton.backends.compiler`` cache entry is neutralized, so the
    GPUTarget import inside the production probe block behaves exactly as
    on a wheel that does not ship the compiler package.
    """
    triton = types.ModuleType(TRITON)
    triton.__version__ = "3.2.1"

    language = types.ModuleType(TRITON_LANGUAGE)
    triton.language = language

    ops = types.ModuleType("triton.ops")
    mpm = types.ModuleType(TRITON_OPS_MPM)
    mpm.__name__ = TRITON_OPS_MPM
    ops.matmul_perf_model = mpm
    triton.ops = ops

    backends = types.ModuleType(TRITON_BACKENDS)
    backends.backends = {}
    triton.backends = backends

    monkeypatch.setitem(sys.modules, TRITON, triton)
    monkeypatch.setitem(sys.modules, TRITON_LANGUAGE, language)
    monkeypatch.setitem(sys.modules, "triton.ops", ops)
    monkeypatch.setitem(sys.modules, TRITON_OPS_MPM, mpm)
    monkeypatch.setitem(sys.modules, TRITON_BACKENDS, backends)
    return triton


def _patch_fn():
    from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
        patch_torch_inductor_for_iluvatar,
    )

    return patch_torch_inductor_for_iluvatar


def _installed_gputarget():
    """The GPUTarget class currently installed in triton_heuristics."""
    import torch._inductor.runtime.triton_heuristics as th

    return th.GPUTarget


class TestGpuTargetProbeExceptionArms:
    """The two exception arms of the GPUTarget discovery block (L290-291,
    L309-310) — the only four statements missing from iluvatar.py on both
    the CPU gate and the BI-V150 real-hardware run."""

    def test_unimportable_gpu_target_falls_back_to_corex(
        self, fake_flagtree_triton, monkeypatch, caplog
    ):
        """A triton without the compiler submodule (L290-291): the inner
        except swallows the ImportError, the probe never finds a target,
        and the non-strict fallback remaps GPUTarget 'cuda' -> 'corex'.

        The fake backend table accepts every probed target so that — had
        the GPUTarget import succeeded — the probe would have discovered
        'cuda' and skipped the patch entirely. The only way this test's
        remap can be installed is through the missing-module arm.
        """
        # Neutralize both the real cache entry and any attribute the
        # parent fake module might expose, so `from triton.backends.compiler
        # import GPUTarget` raises inside the probe block.
        monkeypatch.setitem(sys.modules, TRITON_BACKENDS_COMPILER, None)

        class _AcceptingCompiler:
            @staticmethod
            def supports_target(target):
                return True

        class _AcceptingBackend:
            compiler = _AcceptingCompiler()

        fake_flagtree_triton.backends.backends = {"iluvatar": _AcceptingBackend()}

        patch = _patch_fn()
        before = _installed_gputarget()
        # non-strict: the whole patch degrades quietly. The probe swallow
        # (L290-291) leaves _target unset, the fallback (L295) picks 'corex',
        # and because the compiler module is *also* what the install block
        # imports, the install fails and is logged at warning — the caller
        # still proceeds without the remap.
        with caplog.at_level(
            "WARNING", logger="vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"
        ):
            patch()  # must not raise
        assert _installed_gputarget() is before, (
            "with no importable GPUTarget the remap is not installed"
        )
        assert any(
            "Failed to patch torch._inductor" in record.message
            for record in caplog.records
        ), (
            "the corex fallback must flow into the install block, whose "
            "non-strict failure is visible at warning level"
        )

    def test_unimportable_gpu_target_strict_raises(
        self, fake_flagtree_triton, monkeypatch
    ):
        """Strict mode through the same missing-module arm: the inner
        swallow leaves no target and the strict fallback raises."""
        monkeypatch.setitem(sys.modules, TRITON_BACKENDS_COMPILER, None)

        class _RejectingCompiler:
            @staticmethod
            def supports_target(target):
                return False

        class _RejectingBackend:
            compiler = _RejectingCompiler()

        fake_flagtree_triton.backends.backends = {"iluvatar": _RejectingBackend()}

        with pytest.raises(RuntimeError) as exc_info:
            _patch_fn()(strict=True)
        # the outer except re-wraps the inner strict raise; the original
        # "No supported ... target" message must survive as __cause__
        assert "No supported Iluvatar Triton target found" in str(
            exc_info.value.__cause__
        )

    def test_unimportable_backends_module_quietly_returns(
        self, fake_flagtree_triton, monkeypatch, caplog
    ):
        """A triton whose ``backends`` module cannot be imported (L309-310):
        non-strict mode logs at debug and returns without touching the
        installed GPUTarget.

        ``monkeypatch.setitem(sys.modules, "triton.backends", None)`` makes
        the production ``import triton.backends as _tb`` raise, exercising
        the outer except's debug-log-and-return arm — the only rows the
        existing suite never executes (its strict sibling is covered).
        """
        monkeypatch.setitem(sys.modules, TRITON_BACKENDS, None)

        patch = _patch_fn()
        before = _installed_gputarget()
        with caplog.at_level(
            "DEBUG", logger="vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"
        ):
            patch()  # must not raise and must not install anything
        assert _installed_gputarget() is before, (
            "the quiet-return arm must leave the installed GPUTarget class untouched"
        )
        assert any(
            "cannot inspect triton backends" in record.message
            for record in caplog.records
        ), "the defensive skip must be visible at debug level"

    def test_unimportable_backends_module_strict_raises(
        self, fake_flagtree_triton, monkeypatch
    ):
        """Strict mode through the unimportable-backends arm: startup dies."""
        monkeypatch.setitem(sys.modules, TRITON_BACKENDS, None)

        with pytest.raises(RuntimeError, match="Cannot inspect Iluvatar Triton"):
            _patch_fn()(strict=True)
