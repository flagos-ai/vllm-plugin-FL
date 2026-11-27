# Copyright 2026 BAAI. All rights reserved.
"""CPU coverage for the iluvatar triton/inductor patch functions and the
flaggems activation custom op.

``vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`` runs three triton
patches plus the sampler and inductor patches at import time. Their *success*
branches only execute with a real flagtree triton install, so the coverage
combined JSON reports ~99 missing statements across the five function bodies:
``patch_triton_chained_or_for_iluvatar`` (find_spec / regex rewrite / marker
idempotency / pycache removal / sys.modules eviction), the
``make_tensor_descriptor`` stub, the matmul perf-model clock replacement, the
``__wrapped__`` unwrapping of the sampler compile closures, and the GPUTarget
'cuda' -> 'corex' remap installed into torch._inductor.

These tests drive each function with fake module objects injected into
``sys.modules`` (or a fake source file under tmp_path for the pure file
rewrite), so the patch bodies execute on CPU and their observable effects —
rewritten file content, installed attributes, replaced functions, remapped
GPUTarget backends — are asserted directly.

The last class covers ``vendor/iluvatar/activation.py`` (9 statements, 0%
covered): the ``vllm_fl::iluvatar_gems_silu_and_mul`` custom op whose body
lazily imports flaggems_vllm, executed against a fake flaggems module with an
independent sigmoid oracle.

No monkeypatch of the functions under test themselves: only the *external*
boundary modules (``triton`` / ``vllm.v1.sample.ops.*`` / ``flaggems_vllm`` /
the helper module's file location) are faked, which is exactly what the
production code consumes.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pathlib
import sys
import types

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

# Import the production module NOW, against the real installed triton, so the
# module-level patch calls (lines 207-209/241/346 of iluvatar.py) run against
# the real environment before any test installs a fake triton into
# sys.modules. Without this, the first test's fake triton would drive the
# module-level chained-or rewrite at the real vllm helper file.
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar as _prod_module  # noqa: E402

TRITON = "triton"
TRITON_LANGUAGE = "triton.language"
TRITON_OPS_MPM = "triton.ops.matmul_perf_model"
HELPERS = "vllm.v1.attention.ops.triton_attention_helpers"
TOPK_SAMPLER = "vllm.v1.sample.ops.topk_topp_sampler"
LOGPROBS = "vllm.v1.sample.ops.logprobs"


@pytest.fixture
def fake_triton(monkeypatch):
    """Install a fake ``triton`` (version 3.2.1) into sys.modules.

    The production patches read ``triton.__version__`` to gate on triton
    < 3.3; the fake reports 3.2.1 so the patch bodies run. Everything is
    restored by monkeypatch on teardown.
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

    backends = types.ModuleType("triton.backends")
    triton.backends = backends
    return triton, language, mpm


@pytest.fixture
def install_triton(monkeypatch, fake_triton):
    """Actually install the fake triton tree into sys.modules."""
    triton, language, mpm = fake_triton
    monkeypatch.setitem(sys.modules, TRITON, triton)
    monkeypatch.setitem(sys.modules, TRITON_LANGUAGE, language)
    monkeypatch.setitem(sys.modules, "triton.ops", triton.ops)
    monkeypatch.setitem(sys.modules, TRITON_OPS_MPM, mpm)
    monkeypatch.setitem(sys.modules, "triton.backends", triton.backends)
    return fake_triton


class TestPatchTritonLanguage:
    """``patch_triton_language_for_iluvatar`` installs a descriptor stub."""

    def test_stub_installed_on_triton_lt_3_3(self, install_triton):
        triton, language, _ = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_language_for_iluvatar,
        )

        assert not hasattr(language, "make_tensor_descriptor")
        patch_triton_language_for_iluvatar()
        stub = language.make_tensor_descriptor
        assert stub is not None, "stub must be installed for triton < 3.3"

    def test_stub_is_hashable_and_reproducible(self, install_triton):
        _, language, _ = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_language_for_iluvatar,
        )

        patch_triton_language_for_iluvatar()
        stub = language.make_tensor_descriptor
        # DependencyFinder hashes the attribute: the stub must be hashable.
        assert hash(stub) == hash(stub), "hash must be stable"
        assert "make_tensor_descriptor" in repr(stub)

    def test_stub_call_raises_with_actionable_message(self, install_triton):
        _, language, _ = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_language_for_iluvatar,
        )

        patch_triton_language_for_iluvatar()
        with pytest.raises(RuntimeError, match="USE_TD=0"):
            language.make_tensor_descriptor()
        # calling with args must not blow up differently
        with pytest.raises(RuntimeError):
            language.make_tensor_descriptor(1, 2, kw=3)

    def test_existing_descriptor_is_not_replaced(self, install_triton):
        _, language, _ = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_language_for_iluvatar,
        )

        sentinel = object()
        language.make_tensor_descriptor = sentinel
        patch_triton_language_for_iluvatar()
        assert language.make_tensor_descriptor is sentinel, (
            "triton >= 3.3 (attribute already present) must be a no-op"
        )

    def test_missing_triton_is_swallowed(self, monkeypatch, caplog):
        """A poisoned triton import must log a warning, not raise."""
        import logging

        monkeypatch.setitem(sys.modules, TRITON_LANGUAGE, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_language_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_triton_language_for_iluvatar()  # must not raise
        assert any(
            "triton.language" in r.message for r in caplog.records
        ), "the failure must be logged"


class TestPatchTritonPerfModel:
    """``patch_triton_perf_model_for_iluvatar`` replaces the clock probe."""

    def test_get_clock_rate_replaced_with_fixed_value(self, install_triton):
        _, _, mpm = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_perf_model_for_iluvatar,
        )

        calls = []

        def crashing_probe():
            calls.append("probe")
            raise RuntimeError("neither ixsmi nor nvml")

        mpm.get_clock_rate_in_khz = crashing_probe
        patch_triton_perf_model_for_iluvatar()
        # The patched function must no longer call the crashing probe.
        assert mpm.get_clock_rate_in_khz() == 1500 * 1e3
        assert calls == [], "the nvsmi/nvml probe must be fully bypassed"
        assert mpm.get_clock_rate_in_khz() == mpm.get_clock_rate_in_khz()

    def test_idempotent_marker_prevents_double_patch(self, install_triton):
        _, _, mpm = install_triton
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_perf_model_for_iluvatar,
        )

        patch_triton_perf_model_for_iluvatar()
        first = mpm.get_clock_rate_in_khz
        patch_triton_perf_model_for_iluvatar()
        assert mpm.get_clock_rate_in_khz is first, "second run must no-op"
        assert mpm._iluvatar_clock_patched is True

    def test_triton_3_3_plus_skips_early(self, install_triton, caplog):
        """triton >= 3.3 returns before importing the perf-model module."""
        import logging

        triton, _, _ = install_triton
        triton.__version__ = "3.3.0"
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_perf_model_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_triton_perf_model_for_iluvatar()
        assert not caplog.records, (
            "triton >= 3.3 must return cleanly without touching the except branch"
        )

    def test_missing_perf_model_module_is_swallowed(self, monkeypatch, caplog):
        """triton present but ops.matmul_perf_model absent must not raise."""
        import logging

        triton = types.ModuleType(TRITON)
        triton.__version__ = "3.2.1"
        monkeypatch.setitem(sys.modules, TRITON, triton)
        monkeypatch.setitem(sys.modules, TRITON_OPS_MPM, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_perf_model_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_triton_perf_model_for_iluvatar()  # must not raise
        assert any(
            "perf model" in r.message for r in caplog.records
        ), "the failure must be logged"


class TestPatchTritonChainedOr:
    """``patch_triton_chained_or_for_iluvatar`` rewrites the helper file."""

    @pytest.fixture
    def helpers_dir(self, tmp_path, monkeypatch, install_triton):
        r"""Point find_spec at a fake helper module file under tmp_path.

        The fake source mirrors the real upstream pattern the production
        patch targets: a chained boolean or whose operands are bare word
        identifiers (the regex ``_OPERAND`` only matches ``\w+`` /
        parenthesized groups / ``not X``).
        """
        src = (
            "# fake triton_attention_helpers for chained-or testing\n"
            "def _guard(USE_MM_PREFIX, USE_R_SWA, USE_PER_SEQ_CAUSAL, USE_CAUSAL):\n"
            "    if USE_MM_PREFIX or USE_R_SWA or USE_PER_SEQ_CAUSAL:\n"
            "        return 1\n"
            "    return 0\n"
        )
        helpers = tmp_path / "triton_attention_helpers.py"
        helpers.write_text(src)
        # __pycache__ to prove it gets removed after a rewrite
        pycache = helpers.parent / "__pycache__"
        pycache.mkdir()
        (pycache / "triton_attention_helpers.cpython-312.pyc").write_bytes(b"\x00")

        spec = importlib.machinery.ModuleSpec(
            HELPERS, None, origin=str(helpers), is_package=False
        )
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return spec
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        # also pretend the module was already imported so eviction runs
        monkeypatch.setitem(sys.modules, HELPERS, types.ModuleType(HELPERS))
        return helpers

    def test_rewrites_chained_or_expression(self, helpers_dir):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()
        rewritten = helpers_dir.read_text()
        assert "(USE_MM_PREFIX or USE_R_SWA) or USE_PER_SEQ_CAUSAL" in rewritten, (
            "A or B or C must become (A or B) or C"
        )
        # the patch must remain semantics-preserving Python
        compile(rewritten, str(helpers_dir), "exec")
        # marker appended for idempotency
        assert "# _iluvatar_chained_or_patched" in rewritten

    def test_pycache_removed_and_module_evicted(self, helpers_dir):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        assert (helpers_dir.parent / "__pycache__").exists()
        patch_triton_chained_or_for_iluvatar()
        assert not (helpers_dir.parent / "__pycache__").exists(), (
            "stale bytecode must not shadow the patched source"
        )
        assert HELPERS not in sys.modules, "module must be evicted for reimport"

    def test_pycache_removal_failure_is_non_fatal(
        self, helpers_dir, monkeypatch
    ):
        """A failing __pycache__ removal must not fail the patch (L149-150)."""
        import shutil

        def failing_rmtree(path, *args, **kwargs):
            raise OSError("cannot remove pycache")

        monkeypatch.setattr(shutil, "rmtree", failing_rmtree)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()  # must not raise
        # the rewrite itself still completed: marker present, module evicted
        assert "# _iluvatar_chained_or_patched" in helpers_dir.read_text()
        assert HELPERS not in sys.modules

    def test_second_run_is_idempotent(self, helpers_dir):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()
        after_first = helpers_dir.read_text()
        patch_triton_chained_or_for_iluvatar()
        assert helpers_dir.read_text() == after_first, "marker must stop reruns"

    def test_triton_3_3_plus_is_a_noop(self, helpers_dir, install_triton):
        triton, _, _ = install_triton
        triton.__version__ = "3.3.2"
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        original = helpers_dir.read_text()
        patch_triton_chained_or_for_iluvatar()
        assert helpers_dir.read_text() == original
        assert HELPERS in sys.modules, "no eviction for triton >= 3.3"

    def test_missing_helpers_module_is_skipped(self, monkeypatch, install_triton):
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return None
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()  # must not raise

    def test_unparsable_version_falls_back_defensively(
        self, monkeypatch, tmp_path
    ):
        """A version string that fails int() must still apply the patch."""
        triton = types.ModuleType(TRITON)
        triton.__version__ = "flagtree-3.2.x+corex"  # int("flagtree") raises
        monkeypatch.setitem(sys.modules, TRITON, triton)

        src = "def f(x):\n    return x or 1 or 2\n"
        helpers = tmp_path / "triton_attention_helpers.py"
        helpers.write_text(src)
        spec = importlib.machinery.ModuleSpec(
            HELPERS, None, origin=str(helpers), is_package=False
        )
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return spec
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setitem(sys.modules, HELPERS, types.ModuleType(HELPERS))
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()
        rewritten = helpers.read_text()
        assert "(x or 1) or 2" in rewritten, "defensive path must still rewrite"
        assert "# _iluvatar_chained_or_patched" in rewritten

    def test_unreadable_file_logs_and_skips(
        self, monkeypatch, tmp_path, caplog, install_triton
    ):
        """A read failure must warn and leave everything untouched."""
        import logging

        helpers = tmp_path / "triton_attention_helpers.py"
        helpers.write_text("def f():\n    pass\n")
        spec = importlib.machinery.ModuleSpec(
            HELPERS, None, origin=str(helpers), is_package=False
        )
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return spec
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

        def failing_read(self, *args, **kwargs):
            raise OSError("permission denied reading fake module")

        monkeypatch.setattr(pathlib.Path, "read_text", failing_read)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_triton_chained_or_for_iluvatar()
        assert any("cannot read" in r.message for r in caplog.records)

    def test_no_chained_or_expression_leaves_file_untouched(
        self, monkeypatch, tmp_path, install_triton
    ):
        """A clean file gains neither a rewrite nor the marker."""
        helpers = tmp_path / "triton_attention_helpers.py"
        helpers.write_text("def f(x):\n    return x + 1\n")
        spec = importlib.machinery.ModuleSpec(
            HELPERS, None, origin=str(helpers), is_package=False
        )
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return spec
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setitem(sys.modules, HELPERS, types.ModuleType(HELPERS))
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        patch_triton_chained_or_for_iluvatar()
        assert helpers.read_text() == "def f(x):\n    return x + 1\n"
        assert HELPERS in sys.modules, "nothing evicted when nothing changed"

    def test_unwritable_file_logs_and_skips(
        self, monkeypatch, tmp_path, caplog, install_triton
    ):
        """A write failure must warn and skip the pycache/sys.modules steps."""
        import logging
        import pathlib as _pl

        helpers = tmp_path / "triton_attention_helpers.py"
        helpers.write_text("def f(x):\n    return x or 1 or 2\n")
        spec = importlib.machinery.ModuleSpec(
            HELPERS, None, origin=str(helpers), is_package=False
        )
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == HELPERS:
                return spec
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        helper_module = types.ModuleType(HELPERS)
        monkeypatch.setitem(sys.modules, HELPERS, helper_module)

        original_write = _pl.Path.write_text

        def failing_write(self, data, *a, **kw):
            if self.name == "triton_attention_helpers.py":
                raise OSError("disk full")
            return original_write(self, data, *a, **kw)

        monkeypatch.setattr(_pl.Path, "write_text", failing_write)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_triton_chained_or_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_triton_chained_or_for_iluvatar()
        assert any("cannot write" in r.message for r in caplog.records)
        # skipped the post-write steps: no eviction
        assert sys.modules.get(HELPERS) is helper_module
        assert helpers.read_text() == "def f(x):\n    return x or 1 or 2\n"


class TestPatchSamplerCompile:
    """``patch_sampler_compile_for_iluvatar`` unwraps torch.compile."""

    @staticmethod
    def _fake_sampler_module(wrapped: bool):
        """A fake vllm sampler module whose function is a compile wrapper."""
        module = types.ModuleType(TOPK_SAMPLER)
        module.__name__ = TOPK_SAMPLER

        def original_fn(*args, **kwargs):
            return ("original", args)

        def inner_fn(*args, **kwargs):
            return ("compiled", args)

        # torch.compile exposes the inner function as __wrapped__
        if wrapped:
            inner_fn.__wrapped__ = original_fn
            module.compiled_random_sample = inner_fn
        else:
            module.compiled_random_sample = original_fn
        return module, original_fn

    def test_unwraps_compile_shim_to_original(self, monkeypatch):
        module, original = self._fake_sampler_module(wrapped=True)
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, module)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)  # block second block
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        patch_sampler_compile_for_iluvatar()
        assert module.compiled_random_sample is original, (
            "the torch.compile wrapper must be replaced by its __wrapped__"
        )
        assert getattr(module, "_iluvatar_compile_patched", False)
        assert module.compiled_random_sample(1) == ("original", (1,))

    def test_plain_function_stays_itself_and_sets_marker(self, monkeypatch):
        module, original = self._fake_sampler_module(wrapped=False)
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, module)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        patch_sampler_compile_for_iluvatar()
        # no __wrapped__ -> getattr falls back to the function itself
        assert module.compiled_random_sample is original
        assert getattr(module, "_iluvatar_compile_patched", False)

    def test_already_patched_module_is_untouched(self, monkeypatch):
        module, original = self._fake_sampler_module(wrapped=True)
        module._iluvatar_compile_patched = True
        before = module.compiled_random_sample
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, module)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        patch_sampler_compile_for_iluvatar()
        assert module.compiled_random_sample is before, (
            "the marker must make re-patching a no-op"
        )

    def test_strict_failure_raises_runtime_error(self, monkeypatch):
        """strict=True turns a failed sampler import into a startup error."""
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, None)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        with pytest.raises(RuntimeError, match="topk/topp sampler patch failed"):
            patch_sampler_compile_for_iluvatar(strict=True)

    def test_strict_logprobs_failure_raises_runtime_error(self, monkeypatch):
        """The logprobs block has its own strict failure mode.

        The topk block must succeed first (module already carries the
        marker), then the poisoned logprobs import trips the second strict
        raise.
        """
        topk = types.ModuleType(TOPK_SAMPLER)
        topk._iluvatar_compile_patched = True  # topk block no-ops
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, topk)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        with pytest.raises(RuntimeError, match="logprobs sampler patch failed"):
            patch_sampler_compile_for_iluvatar(strict=True)

    def test_non_strict_failure_only_warns(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, TOPK_SAMPLER, None)
        monkeypatch.setitem(sys.modules, LOGPROBS, None)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_sampler_compile_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_sampler_compile_for_iluvatar()  # must not raise
        assert any(
            "topk_topp" in r.message for r in caplog.records
        ), "failure must be logged, not silent"


class TestPatchTorchInductor:
    """``patch_torch_inductor_for_iluvatar`` remaps 'cuda' GPUTargets."""

    @pytest.fixture(autouse=True)
    def restore_triton_heuristics(self):
        """Undo any GPUTarget installation after each test.

        The patch writes ``GPUTarget`` + a module marker into the real
        ``torch._inductor.runtime.triton_heuristics``; without this teardown
        the installation would leak into every later test in the process.
        """
        import torch._inductor.runtime.triton_heuristics as th
        from triton.backends.compiler import GPUTarget as _Orig

        yield
        th.GPUTarget = _Orig
        if hasattr(th, "_iluvatar_gputarget_patched"):
            del th._iluvatar_gputarget_patched

    @pytest.fixture
    def corex_only_backends(self, monkeypatch):
        """Fake a flagtree triton whose only backend accepts 'corex'."""
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {"iluvatar": _fake_corex_backend()})
        return tb

    @staticmethod
    def _installed_gputarget():
        import torch._inductor.runtime.triton_heuristics as th

        return th.GPUTarget

    def test_gputarget_remapped_cuda_to_corex(self, corex_only_backends):
        from triton.backends.compiler import GPUTarget

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        before = self._installed_gputarget()
        patch_torch_inductor_for_iluvatar()
        installed = self._installed_gputarget()
        assert installed is not before, "GPUTarget must be replaced"
        # constructing a 'cuda' target now yields the remapped backend
        target = installed("cuda", 90, False)
        assert target.backend == "corex", (
            "'cuda' must be remapped to the flagtree backend target"
        )
        # non-cuda backends pass through untouched
        assert installed("other", 90, False).backend == "other"
        # the installed class is a GPUTarget subclass (isinstance survives)
        assert isinstance(target, GPUTarget)

    def test_patch_is_idempotent(self, corex_only_backends):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        patch_torch_inductor_for_iluvatar()
        first = self._installed_gputarget()
        patch_torch_inductor_for_iluvatar()
        assert self._installed_gputarget() is first, "marker must stop re-install"
        assert first("cuda", 90, False).backend == "corex"

    def test_cuda_backend_already_registered_skips(self, monkeypatch):
        """A triton with a 'cuda' backend needs no remap at all."""
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {"cuda": object()})
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        before = self._installed_gputarget()
        patch_torch_inductor_for_iluvatar()
        assert self._installed_gputarget() is before, (
            "'cuda' already registered must leave GPUTarget alone"
        )

    def test_no_backends_registered_debug_skip(self, monkeypatch):
        """No backends at all: the patch returns early (non-strict)."""
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {})
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        before = self._installed_gputarget()
        patch_torch_inductor_for_iluvatar()
        assert self._installed_gputarget() is before

    def test_strict_no_backends_raises(self, monkeypatch):
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {})
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        # the outer except re-raises as "Cannot inspect ...", chaining the
        # inner "No Triton backend registered" as __cause__
        with pytest.raises(RuntimeError, match="Cannot inspect Iluvatar Triton"):
            patch_torch_inductor_for_iluvatar(strict=True)

    def test_strict_unsupportable_target_raises(self, monkeypatch):
        """No backend accepts any probed target: strict mode must fail loudly."""
        import triton.backends as tb

        monkeypatch.setattr(
            tb,
            "backends",
            {"iluvatar": _fake_rejecting_backend()},
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        with pytest.raises(RuntimeError) as exc_info:
            patch_torch_inductor_for_iluvatar(strict=True)
        # the strict "No supported ... target" raise is re-wrapped by the
        # outer except; the original message must survive as __cause__
        assert "No supported Iluvatar Triton target found" in str(
            exc_info.value.__cause__
        )

    def test_non_strict_unsupportable_target_falls_back_to_corex(self, monkeypatch):
        import triton.backends as tb

        monkeypatch.setattr(
            tb,
            "backends",
            {"iluvatar": _fake_rejecting_backend()},
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        patch_torch_inductor_for_iluvatar()
        installed = self._installed_gputarget()
        assert installed("cuda", 90, False).backend == "corex", (
            "non-strict fallback must remap to the compatibility target"
        )

    def test_probe_exception_is_swallowed_and_next_probe_wins(
        self, monkeypatch
    ):
        """A backend whose supports_target raises must not abort the probe."""
        import triton.backends as tb

        class _ExplodingCompiler:
            @staticmethod
            def supports_target(target):
                raise RuntimeError("backend exploded during probe")

        class _Exploding:
            compiler = _ExplodingCompiler()

        monkeypatch.setattr(
            tb, "backends", {"bad": _Exploding(), "corex": _fake_corex_backend()}
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        patch_torch_inductor_for_iluvatar()
        installed = self._installed_gputarget()
        assert installed("cuda", 90, False).backend == "corex", (
            "the exploding backend must be skipped, not fatal"
        )

    def test_strict_install_failure_raises_inductor_patch_error(self, monkeypatch):
        """A failing triton_heuristics install fails strictly (L339-342).

        The probe block (first try) succeeds — the corex backend is
        discovered — but the install block binds ``_th`` to a broken
        object (the parent package attribute and sys.modules entry are
        both replaced), so setting ``_th.GPUTarget`` raises and strict
        mode converts it into 'Iluvatar Inductor target patch failed'.
        """
        import torch._inductor.runtime as _runtime
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {"iluvatar": _fake_corex_backend()})
        # int: attribute *writes* raise; getattr falls back to the default.
        # Both the sys.modules entry and the parent-package attribute are
        # replaced, because `import a.b.c as x` re-resolves through the
        # parent when the sys.modules entry is not a module.
        monkeypatch.setitem(
            sys.modules, "torch._inductor.runtime.triton_heuristics", 5
        )
        monkeypatch.setattr(_runtime, "triton_heuristics", 5)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        with pytest.raises(RuntimeError, match="Inductor target patch failed"):
            patch_torch_inductor_for_iluvatar(strict=True)

    def test_non_strict_install_failure_only_warns(self, monkeypatch, caplog):
        """Same install failure without strict only logs a warning."""
        import logging

        import torch._inductor.runtime as _runtime
        import triton.backends as tb

        monkeypatch.setattr(tb, "backends", {"iluvatar": _fake_corex_backend()})
        monkeypatch.setitem(
            sys.modules, "torch._inductor.runtime.triton_heuristics", 5
        )
        monkeypatch.setattr(_runtime, "triton_heuristics", 5)
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_torch_inductor_for_iluvatar,
        )

        with caplog.at_level(logging.WARNING, logger="vllm_fl"):
            patch_torch_inductor_for_iluvatar()  # must not raise
        assert any(
            "Failed to patch torch._inductor" in r.message for r in caplog.records
        )


class TestWorkerPatchHelpers:
    """The iluvatar.py worker-patch helper bodies (A-class, ~27 rows).

    * ``_release_benchmark_tensors`` wraps LibTuner.run so per-shape replay
      arguments are dropped in a finally block.
    * ``patch_autotune_memory`` installs that wrapper once (marker).
    * ``_is_uva_available`` pins False.
    * ``patch_v2_uva`` gates on the platform vendor and replaces
      ``buffer_utils.is_uva_available`` exactly once.
    """

    def test_release_benchmark_tensors_clears_replay_args_in_finally(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _release_benchmark_tensors,
        )

        seen = []
        tuner_self = SimpleNamespace()

        def run(self, *args, **kwargs):
            seen.append(("run", args))
            # simulate the tuner recording replay state mid-run
            self._last_benchmark_args = (args,)
            self._last_benchmark_meta = {"a": 1}
            self.nargs = args
            return "result"

        wrapped = _release_benchmark_tensors(run)
        out = wrapped(tuner_self, 1, 2, key=3)
        assert out == "result", "the wrapper returns the run result"
        assert seen == [("run", (1, 2))]
        # finally block cleared the replay state
        assert tuner_self._last_benchmark_args == ()
        assert tuner_self._last_benchmark_meta == {}
        assert tuner_self.nargs is None

    def test_release_benchmark_tensors_clears_even_when_run_raises(self):
        from types import SimpleNamespace

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _release_benchmark_tensors,
        )

        tuner_self = SimpleNamespace()

        def run(self, *args, **kwargs):
            self._last_benchmark_args = (1,)
            self._last_benchmark_meta = {"a": 1}
            self.nargs = (1,)
            raise RuntimeError("tuner blew up")

        wrapped = _release_benchmark_tensors(run)
        with pytest.raises(RuntimeError, match="tuner blew up"):
            wrapped(tuner_self)
        assert tuner_self._last_benchmark_args == ()
        assert tuner_self._last_benchmark_meta == {}
        assert tuner_self.nargs is None, "finally runs even after a failure"

    def test_release_benchmark_tensors_is_idempotent(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _release_benchmark_tensors,
        )

        def run(self, *args, **kwargs):
            return None

        once = _release_benchmark_tensors(run)
        assert _release_benchmark_tensors(once) is once, (
            "the marker must prevent double wrapping"
        )
        assert getattr(once, "_iluvatar_release_benchmark_tensors", False)

    def test_patch_autotune_memory_wraps_libtuner_run_once(self, monkeypatch):
        import importlib

        # flag_gems.utils re-exports a `libentry` *function* that shadows
        # the submodule name in `import flag_gems.utils.libentry`; the
        # module itself must be resolved through importlib.
        libentry_mod = importlib.import_module("flag_gems.utils.libentry")
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_autotune_memory,
        )

        def fake_run(self, *args, **kwargs):
            return None

        monkeypatch.setattr(libentry_mod.LibTuner, "run", fake_run)
        patch_autotune_memory()
        first = libentry_mod.LibTuner.run
        assert getattr(first, "_iluvatar_release_benchmark_tensors", False), (
            "LibTuner.run must carry the release wrapper"
        )
        assert first.__wrapped__ is fake_run
        patch_autotune_memory()
        assert libentry_mod.LibTuner.run is first, "second call is a no-op"

    def test_is_uva_available_is_false(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _is_uva_available,
        )

        assert _is_uva_available() is False

    def test_patch_v2_uva_replaces_buffer_utils_once(self, monkeypatch):
        import vllm.platforms

        from vllm.v1.worker.gpu import buffer_utils

        # patch_v2_uva imports current_platform from vllm.platforms inside
        # its body, so the vendor gate must be faked on that module.
        monkeypatch.setattr(
            vllm.platforms,
            "current_platform",
            _vendor_ns("iluvatar"),
        )
        original = buffer_utils.is_uva_available
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _is_uva_available,
            patch_v2_uva,
        )

        try:
            patch_v2_uva()
            assert buffer_utils.is_uva_available is _is_uva_available
            assert buffer_utils.is_uva_available() is False
            patch_v2_uva()
            assert buffer_utils.is_uva_available is _is_uva_available, (
                "a second patch must not stack"
            )
        finally:
            buffer_utils.is_uva_available = original

    def test_patch_v2_uva_foreign_vendor_is_noop(self, monkeypatch):
        import vllm.platforms

        from vllm.v1.worker.gpu import buffer_utils

        monkeypatch.setattr(
            vllm.platforms,
            "current_platform",
            _vendor_ns("nvidia"),
        )
        original = buffer_utils.is_uva_available
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_v2_uva,
        )

        patch_v2_uva()
        assert buffer_utils.is_uva_available is original, (
            "a foreign vendor must leave buffer_utils untouched"
        )


def _vendor_ns(name):
    return types.SimpleNamespace(vendor_name=name)


class TestActivationCustomOpLazyImport:
    """``vendor/iluvatar/activation.py`` custom op with a fake flaggems.

    The op (0/9 statements in the combined coverage JSON) lazily imports
    ``flaggems_vllm.silu_and_mul`` inside its body — on CPU neither
    flaggems nor the device kernel exists, so a fake module is injected
    into sys.modules and the op is called for real through the torch
    dispatcher. The fake computes the same formula as the production
    delegate (``F.silu(x1) * x2``) via an *independently written* oracle
    here, so the test verifies the slicing/wiring of the op body
    (``x[..., :d]`` / ``x[..., d:]``), not the vendor kernel.
    """

    @pytest.fixture(autouse=True)
    def register_custom_op(self):
        """Import activation.py so the custom op exists in this process."""
        import vllm_fl.dispatch.backends.vendor.iluvatar.activation as act

        assert callable(act._iluvatar_gems_silu_and_mul)
        yield

    def test_custom_op_splits_and_dispatches_to_flaggems(self, monkeypatch):
        calls = []

        def fake_flaggems_silu_and_mul(x1, x2):
            calls.append((x1, x2))
            return torch.nn.functional.silu(x1.double()).to(x1.dtype) * x2

        fake = types.ModuleType("flaggems_vllm")
        fake.silu_and_mul = fake_flaggems_silu_and_mul
        monkeypatch.setitem(sys.modules, "flaggems_vllm", fake)

        x = torch.randn(6, 8, dtype=torch.float32)
        d = x.shape[-1] // 2
        out = torch.ops.vllm_fl.iluvatar_gems_silu_and_mul(x)
        # the op body must slice [.., :d] and [.., d:] before dispatch
        assert len(calls) == 1
        seen_x1, seen_x2 = calls[0]
        assert torch.equal(seen_x1, x[..., :d])
        assert torch.equal(seen_x2, x[..., d:])
        # independent oracle: sigmoid formula written in the test
        x1, x2 = x[..., :d], x[..., d:]
        oracle = (x1 * torch.sigmoid(x1)) * x2
        assert torch.allclose(out, oracle), "output must match the silu oracle"
        assert out.shape == (6, 4)
        assert out.dtype == x.dtype

    def test_custom_op_registered_in_torch_namespace(self):
        assert hasattr(torch.ops.vllm_fl, "iluvatar_gems_silu_and_mul")

    def test_fake_kernel_meta_shape_only(self):
        """The register_fake meta kernel derives the output shape on CPU."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.randn(3, 20, dtype=torch.bfloat16)
            out = torch.ops.vllm_fl.iluvatar_gems_silu_and_mul(fake_x)
            assert out.shape == (3, 10)
            assert out.dtype == torch.bfloat16


def _fake_corex_backend():
    """A triton backend whose compiler accepts only target 'corex'."""

    class _Compiler:
        @staticmethod
        def supports_target(target):
            return target.backend == "corex"

    class _Backend:
        compiler = _Compiler()

    return _Backend()


def _fake_rejecting_backend():
    """A triton backend that accepts no target (probes all fail)."""

    class _Compiler:
        @staticmethod
        def supports_target(target):
            return False

    class _Backend:
        compiler = _Compiler()

    return _Backend()
