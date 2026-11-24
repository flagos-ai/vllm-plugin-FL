# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the remaining ``vllm_fl.patches`` modules.

* ``exponential_compat`` — the ``Tensor.exponential_(generator=...)`` shim:
  generator-aware path against a seeded inverse-CDF oracle, and the vendor
  fallback (kernel without ``generator``) that warns once per process.
* ``flagtree_target_info`` — the AMD architecture predicates added to
  ``triton.language.target_info``: per-arch truth table, non-HIP guards,
  constexpr decoration, caching and the unavailable-triton branch.
* ``fla_module_relocation`` — the vLLM 0.24 → 0.28 FLA path alias: no-op
  when the old package exists, no-op when the new one is absent, and the
  full aliasing (ops subpackage, module attributes) otherwise.
* ``triton_kernel`` — ``KernelLaunchMetaProxy`` launch overrides and
  ``patch_kernel_launch_meta`` idempotence/conflict behaviour.
* ``qsa_cache_hint`` — the AST surgery removing the K-cache eviction hint
  plus the full patch gate matrix.
* ``arm_cpu_gdn`` — the dense CPU state-indices wrapper around the GDN
  builder's ``build``.
* ``minicpmo_audio`` — the MiniCPM-O encoder token budget extension.

Every module is loaded through ``importlib`` against stubbed external
boundaries with ``monkeypatch``-restored ``sys.modules`` entries, so the
tests are order-independent.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_ROOT = Path(__file__).resolve().parents[4]


def _load(source: Path, module_name: str, monkeypatch, **stub_modules):
    """Exec a plugin source file once under a private module name.

    Any ``stub_modules`` mapping is installed into ``sys.modules`` before
    the exec (with ``monkeypatch.setitem`` restoring the previous entries
    on teardown), so the source file's external imports resolve.
    """
    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Minimal vllm.logger stand-in recording level calls."""

    def __init__(self):
        self.records = []

    def _log(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log("info", msg, *args)

    def debug(self, msg, *args):
        self._log("debug", msg, *args)

    def warning(self, msg, *args):
        self._log("warning", msg, *args)

    def info_once(self, msg, *args):
        self._log("info_once", msg, *args)

    def warning_once(self, msg, *args):
        self._log("warning_once", msg, *args)


def _vllm_logger_stub(logger):
    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_logger = ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logger
    return {"vllm": vllm_pkg, "vllm.logger": vllm_logger}


# ===========================================================================
# exponential_compat
# ===========================================================================


@pytest.fixture
def compat(monkeypatch):
    """Load a fresh copy of the exponential_ shim; restore torch state."""
    builtin_method = torch.Tensor.__dict__.get("exponential_")
    if builtin_method is None:  # pragma: no cover - defensive
        builtin_method = torch._C.TensorBase.exponential_

    module = _load(
        _ROOT / "vllm_fl/patches/exponential_compat.py",
        "supp_exponential_compat_under_test",
        monkeypatch,
    )
    yield module
    torch.Tensor.exponential_ = builtin_method


@pytest.fixture
def vendor_kernel(monkeypatch):
    """Install an aten-style exponential_ kernel lacking ``generator``."""
    calls = []

    def kernel(tensor, lambd=1.0):
        calls.append((float(lambd), tensor is not None))
        return tensor

    monkeypatch.setattr(torch.ops.aten, "exponential_", kernel)
    return calls


class TestExponentialCompat:
    def test_generator_path_matches_seeded_oracle(self, compat):
        """With a generator, the shim delegates to aten and matches -log(U)/lambd."""
        assert compat.apply_exponential_compat() is True
        assert compat.apply_exponential_compat() is False, "second install is a no-op"

        generator = torch.Generator(device="cpu").manual_seed(1234)
        drawn = torch.full((6,), 0.5, dtype=torch.float64)
        result = drawn.exponential_(lambd=1.0, generator=generator)
        assert result is drawn, "in-place exponential_ must return the same tensor"

        # Independent oracle: the same generator seed advanced by the same
        # number of uniform draws must satisfy x = -log(1 - U) / lambd
        # elementwise (the documented torch CPU inverse-CDF for exponential_).
        oracle_generator = torch.Generator(device="cpu").manual_seed(1234)
        uniforms = torch.rand(6, generator=oracle_generator, dtype=torch.float64)
        expected = -torch.log(1 - uniforms)
        torch.testing.assert_close(drawn, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("lambd", [1.0, 2.0, 0.5])
    def test_generator_path_respects_lambd(self, compat, lambd):
        """The lambd argument scales the seeded exponential draw exactly."""
        compat.apply_exponential_compat()

        generator = torch.Generator(device="cpu").manual_seed(99)
        drawn = torch.full((8,), 0.5, dtype=torch.float64)
        drawn.exponential_(lambd=lambd, generator=generator)

        oracle_generator = torch.Generator(device="cpu").manual_seed(99)
        uniforms = torch.rand(8, generator=oracle_generator, dtype=torch.float64)
        expected = -torch.log(1 - uniforms) / lambd
        torch.testing.assert_close(drawn, expected, rtol=1e-12, atol=1e-12)

    def test_generator_kwarg_forwarded_to_aten(self, compat, monkeypatch):
        """The generator is passed as a keyword, matching vLLM's TopKTopP usage."""
        compat.apply_exponential_compat()

        seen = {}
        original = torch.ops.aten.exponential_

        def recording_kernel(tensor, lambd=1.0, generator=None):
            seen["lambd"] = lambd
            seen["generator"] = generator
            seen["is_tensor"] = isinstance(tensor, torch.Tensor)
            return tensor

        try:
            monkeypatch.setattr(torch.ops.aten, "exponential_", recording_kernel)
            tensor = torch.full((2,), 0.5, dtype=torch.float64)
            generator = torch.Generator().manual_seed(7)
            tensor.exponential_(lambd=1.0, generator=generator)
            assert seen == {
                "lambd": 1.0,
                "generator": generator,
                "is_tensor": True,
            }
        finally:
            torch.ops.aten.exponential_ = original

    def test_vendor_kernel_falls_back_without_generator(
        self, compat, vendor_kernel, caplog
    ):
        """A vendor kernel without ``generator`` triggers one warned fallback."""
        compat.apply_exponential_compat()

        tensor = torch.full((3,), 0.5, dtype=torch.float64)
        with caplog.at_level(
            logging.WARNING, logger="supp_exponential_compat_under_test"
        ):
            tensor.exponential_(lambd=1.0, generator=torch.Generator().manual_seed(3))

        assert vendor_kernel == [(1.0, True)], "exactly one fallback call"
        assert "ignores the generator" in caplog.text
        assert compat._warned is True

    def test_vendor_fallback_warns_only_once(self, compat, vendor_kernel, caplog):
        """The fallback warning is emitted once per module, not per call."""
        compat.apply_exponential_compat()

        with caplog.at_level(logging.WARNING, logger="supp_exponential_compat_under_test"):
            for _ in range(3):
                tensor = torch.full((2,), 0.5, dtype=torch.float64)
                tensor.exponential_(generator=torch.Generator().manual_seed(1))

        assert len(vendor_kernel) == 3
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "the vendor-kernel warning must fire exactly once"

    def test_no_generator_call_skips_aten_generator_branch(self, compat, vendor_kernel):
        """Without a generator the shim calls aten directly (no try/except path)."""
        compat.apply_exponential_compat()

        tensor = torch.full((4,), 0.5, dtype=torch.float32)
        tensor.exponential_()

        assert vendor_kernel == [(1.0, True)]
        assert compat._warned is False, "no warning without a generator argument"

    @pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (4, 1, 2)])
    def test_shim_preserves_shape_dtype_and_device(self, compat, shape):
        """The shim is a transparent in-place wrapper: shape/dtype/device stable."""
        compat.apply_exponential_compat()

        generator = torch.Generator().manual_seed(11)
        tensor = torch.full(shape, 0.5, dtype=torch.float32)
        result = tensor.exponential_(generator=generator)

        assert result is tensor
        assert result.shape == shape
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"
        assert torch.all(result >= 0), "exponential draws are non-negative"
        assert torch.all(torch.isfinite(result))


# ===========================================================================
# flagtree_target_info
# ===========================================================================


class FakeTargetInfo:
    """FlagTree-like ``triton.language.target_info`` with a mutable arch."""

    def __init__(self, arch="", hip=True):
        self.arch = arch
        self.is_hip_returns = hip
        self.constexpr_decorated = []

    def is_hip(self):
        return self.is_hip_returns

    def current_target(self):
        target = SimpleNamespace(arch=self.arch)
        return target

    def constexpr_function(self, function):
        self.constexpr_decorated.append(function.__name__)
        return function


@pytest.fixture
def flagtree(monkeypatch):
    """Load a fresh copy of the flagtree patch module with vllm.logger stubbed."""
    logger = _FakeLogger()
    module = _load(
        _ROOT / "vllm_fl/patches/flagtree_target_info.py",
        "supp_flagtree_target_info_under_test",
        monkeypatch,
        **_vllm_logger_stub(logger),
    )
    module.logger = logger
    return module


@pytest.fixture
def fake_target_info(monkeypatch):
    """Install a fake triton.language.target_info in sys.modules.

    The patch module imports ``triton.language.target_info`` inside
    ``apply_flagtree_target_info``, so pre-seeding ``sys.modules`` (and the
    parent package attribute) controls what it sees; monkeypatch restores
    the previous entries on teardown. On hosts with a real triton the real
    parent package is reused; without one the parent packages are faked.
    """
    fake = FakeTargetInfo()
    try:
        import triton.language as real_parent  # noqa: F401

        have_real = True
    except ImportError:  # pragma: no cover - hosts without triton
        have_real = False

    if have_real:
        monkeypatch.setitem(sys.modules, "triton.language.target_info", fake)
        real_parent.target_info = fake
    else:
        triton_pkg = ModuleType("triton")
        triton_pkg.__path__ = []
        language_pkg = ModuleType("triton.language")
        language_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "triton", triton_pkg)
        monkeypatch.setitem(sys.modules, "triton.language", language_pkg)
        monkeypatch.setitem(sys.modules, "triton.language.target_info", fake)
    yield fake


_PREDICATES = (
    "is_hip_gfx1250",
    "is_hip_cdna",
    "is_hip_cdna1",
    "is_hip_cdna2",
    "is_hip_rdna",
    "is_hip_gfx11",
    "is_hip_gfx12",
)


class TestFlagtreeTargetInfo:
    @pytest.mark.parametrize(
        "arch,expected_true",
        [
            # Prefixes overlap by design: a gfx1250 arch also satisfies the
            # broader gfx12/rdna families, mirroring Triton's semantics.
            ("gfx1250", {"is_hip_gfx1250", "is_hip_gfx12", "is_hip_rdna"}),
            ("gfx1251", {"is_hip_gfx1250", "is_hip_gfx12", "is_hip_rdna"}),
            ("gfx908", {"is_hip_cdna", "is_hip_cdna1"}),
            ("gfx90a", {"is_hip_cdna", "is_hip_cdna2"}),
            ("gfx942", {"is_hip_cdna"}),
            ("gfx950", {"is_hip_cdna"}),
            ("gfx1100", {"is_hip_rdna", "is_hip_gfx11"}),
            ("gfx1150", {"is_hip_rdna", "is_hip_gfx11"}),
            ("gfx1201", {"is_hip_rdna", "is_hip_gfx12"}),
            ("gfx1030", set()),  # RDNA3 has no predicate in the patch set
        ],
    )
    def test_predicate_truth_table_per_arch(
        self, flagtree, fake_target_info, arch, expected_true
    ):
        """Each arch makes exactly its family's predicates true, others false."""
        fake_target_info.arch = arch

        assert flagtree.apply_flagtree_target_info() is True

        for name in sorted(_PREDICATES):
            predicate = getattr(fake_target_info, name, None)
            assert callable(predicate), f"{name} was not added to target_info"
            expected = name in expected_true
            assert predicate() is expected, (
                f"{name}() for arch={arch}: expected {expected}, got {predicate()}"
            )

    def test_non_hip_target_disables_every_predicate(
        self, flagtree, fake_target_info
    ):
        """A CUDA/CPU target makes every added predicate False even on gfx names."""
        fake_target_info.arch = "gfx942"
        fake_target_info.is_hip_returns = False

        assert flagtree.apply_flagtree_target_info() is True
        for name in _PREDICATES:
            assert getattr(fake_target_info, name)() is False, name

    def test_existing_attributes_are_left_alone(self, flagtree, fake_target_info):
        """Predicates FlagTree already exports are never overwritten."""

        def real_flagtree_gfx1250():
            return "flagtree-original"

        fake_target_info.is_hip_gfx1250 = real_flagtree_gfx1250
        fake_target_info.arch = "gfx908"

        assert flagtree.apply_flagtree_target_info() is True
        assert fake_target_info.is_hip_gfx1250 is real_flagtree_gfx1250
        assert fake_target_info.is_hip_gfx1250() == "flagtree-original"
        # Missing siblings were still added.
        assert callable(fake_target_info.is_hip_cdna)
        assert fake_target_info.is_hip_cdna() is True

    def test_added_predicates_are_decorated_as_constexpr(
        self, flagtree, fake_target_info
    ):
        """When FlagTree offers constexpr_function, added predicates get wrapped."""
        fake_target_info.arch = "gfx908"

        flagtree.apply_flagtree_target_info()

        assert set(fake_target_info.constexpr_decorated) == set(_PREDICATES)

    def test_broken_constexpr_decorator_falls_back_to_plain(
        self, flagtree, fake_target_info
    ):
        """A constexpr_function that raises must not break the patch."""

        def broken(function):
            raise RuntimeError("flagtree constexpr bug")

        fake_target_info.constexpr_function = broken
        fake_target_info.arch = "gfx908"

        assert flagtree.apply_flagtree_target_info() is True
        assert fake_target_info.is_hip_cdna() is True
        assert fake_target_info.is_hip_gfx1250() is False

    def test_unimportable_triton_reports_false(self, flagtree, monkeypatch):
        """When triton.language.target_info cannot import, apply returns False."""
        monkeypatch.setitem(sys.modules, "triton.language.target_info", None)

        assert flagtree.apply_flagtree_target_info() is False
        assert any(level == "debug" for level, _ in flagtree.logger.records)

    def test_current_target_failure_yields_empty_arch(
        self, flagtree, fake_target_info
    ):
        """A raising current_target makes predicates False instead of crashing."""

        def broken_target():
            raise RuntimeError("no target on this stack")

        fake_target_info.current_target = broken_target

        assert flagtree.apply_flagtree_target_info() is True
        for name in ("is_hip_cdna", "is_hip_rdna"):
            assert getattr(fake_target_info, name)() is False, name

    @pytest.mark.parametrize("target_value", [None, object()])
    def test_non_string_arch_is_treated_as_empty(
        self, flagtree, fake_target_info, target_value
    ):
        """A target without a string ``arch`` attribute must not crash predicates."""

        def current_target():
            return target_value

        fake_target_info.current_target = current_target

        assert flagtree.apply_flagtree_target_info() is True
        assert fake_target_info.is_hip_cdna() is False
        assert fake_target_info.is_hip_gfx11() is False

    def test_apply_is_cached_and_logs_added_names(self, flagtree, fake_target_info):
        """Repeat calls return the cached result; the first call logs the adds."""
        fake_target_info.arch = "gfx908"

        first = flagtree.apply_flagtree_target_info()
        second = flagtree.apply_flagtree_target_info()

        assert first is True and second is True
        infos = [message for level, message in flagtree.logger.records if level == "info"]
        assert len(infos) == 1, "info log must be emitted once despite the lru_cache"
        assert "7 architecture predicate(s)" in infos[0]
        assert "is_hip_gfx1250" in infos[0]


# ===========================================================================
# fla_module_relocation
# ===========================================================================


def _fake_pkg(name):
    """A fake package ``find_spec`` resolves (needs __spec__ set).

    ``importlib.util.find_spec`` returns the ``__spec__`` attribute of a
    module already in ``sys.modules``; a bare ``ModuleType`` has ``__spec__
    = None`` which raises ValueError (reported as "does not exist" by the
    patch's ``_module_exists``). Setting a ModuleSpec makes the fake
    indistinguishable from an installed package for the probe.
    """
    import importlib.machinery

    module = ModuleType(name)
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    return module


@pytest.fixture
def fla(monkeypatch):
    """Load the FLA relocation patch module under a private name."""
    spec = importlib.util.spec_from_file_location(
        "supp_fla_relocation_under_test",
        _ROOT / "vllm_fl/patches/fla_module_relocation.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "supp_fla_relocation_under_test", module)
    spec.loader.exec_module(module)
    return module


class TestFlaModuleRelocation:
    _OLD = "vllm.model_executor.layers.fla"
    _NEW = "vllm.third_party.flash_linear_attention"

    def _cleanup_aliases(self):
        """Remove any sys.modules/attribute aliases install() created."""
        sys.modules.pop(self._OLD, None)
        sys.modules.pop(self._OLD + ".ops", None)
        layers = sys.modules.get("vllm.model_executor.layers")
        if layers is not None and hasattr(layers, "fla"):
            try:
                delattr(layers, "fla")
            except AttributeError:  # pragma: no cover - defensive
                pass

    def test_noop_when_old_package_exists(self, fla, monkeypatch):
        """vLLM <= 0.24 ships the real ...layers.fla: leave it alone."""
        monkeypatch.setitem(sys.modules, self._OLD, _fake_pkg(self._OLD))
        try:
            assert fla.install() is False
        finally:
            self._cleanup_aliases()

    def test_noop_when_new_package_absent(self, fla, monkeypatch):
        """Without the >= 0.28 package there is nothing to alias."""
        layers = _fake_pkg("vllm.model_executor.layers")
        monkeypatch.setitem(sys.modules, "vllm", _fake_pkg("vllm"))
        monkeypatch.setitem(sys.modules, "vllm.model_executor", _fake_pkg("vllm.model_executor"))
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)
        # The old prefix is genuinely absent (parent has an empty path so the
        # finders cannot locate a child) and the new prefix is absent too.
        assert fla.install() is False

    def test_aliases_new_package_under_old_prefix(self, fla, monkeypatch):
        """The new package and its ops subpackage are aliased."""
        # vLLM 0.32 ships BOTH the old layers.fla package and the new
        # third_party one, so find_spec would short-circuit install() to
        # its "old package exists" no-op. Pin the existence probes so the
        # aliasing branch runs regardless of what this build ships.
        monkeypatch.setattr(fla, "_module_exists", lambda name: name != self._OLD)
        new_pkg = _fake_pkg(self._NEW)
        ops_pkg = _fake_pkg(self._NEW + ".ops")
        layers_pkg = _fake_pkg("vllm.model_executor.layers")
        monkeypatch.setitem(sys.modules, "vllm", _fake_pkg("vllm"))
        monkeypatch.setitem(sys.modules, "vllm.third_party", _fake_pkg("vllm.third_party"))
        monkeypatch.setitem(sys.modules, self._NEW, new_pkg)
        monkeypatch.setitem(sys.modules, self._NEW + ".ops", ops_pkg)
        monkeypatch.setitem(sys.modules, "vllm.model_executor", _fake_pkg("vllm.model_executor"))
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers_pkg)
        # A live session may already carry old-prefix alias entries (the
        # kunlunxin plugin installs them at vllm import). install() is a
        # no-op while the old prefix resolves, so stash and remove them for
        # the duration of this test, then restore.
        stash = {}
        for name in list(sys.modules):
            if name.startswith(self._OLD):
                stash[name] = sys.modules.pop(name)

        try:
            assert fla.install() is True
            assert sys.modules[self._OLD + ".ops"] is ops_pkg
            assert layers_pkg.fla is new_pkg
            # The bare old prefix aliases the new package root.
            assert sys.modules[self._OLD] is new_pkg
        finally:
            self._cleanup_aliases()
            sys.modules.update(stash)


# ===========================================================================
# triton_kernel (KernelLaunchMetaProxy)
# ===========================================================================


class TestKernelLaunchMetaProxy:
    def _kernel(self):
        calls = []

        def launch(*args, **kwargs):
            calls.append((args, kwargs))
            return "launched"

        class Kernel:
            def __getitem__(self, grid):
                return launch

            attribute = "kernel-attribute"

        return Kernel(), calls

    def test_overrides_are_applied_at_launch(self):
        """kwargs overrides reach the underlying launch."""
        from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy

        kernel, calls = self._kernel()
        proxy = KernelLaunchMetaProxy(kernel, {"num_warps": 4})

        result = proxy[(1, 2)]("a", b="b")

        assert result == "launched"
        assert calls == [(("a",), {"b": "b", "num_warps": 4})]

    def test_launch_overrides_property_returns_copy(self):
        """The public property returns a copy, not the internal dict."""
        from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy

        kernel, _ = self._kernel()
        proxy = KernelLaunchMetaProxy(kernel, {"num_warps": 4})
        overrides = proxy.launch_overrides
        overrides["num_warps"] = 8
        assert proxy.launch_overrides == {"num_warps": 4}

    def test_empty_overrides_rejected(self):
        """At least one override is required."""
        from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy

        kernel, _ = self._kernel()
        with pytest.raises(ValueError, match="At least one"):
            KernelLaunchMetaProxy(kernel, {})

    def test_getattr_delegates_to_kernel(self):
        """Unknown attributes fall through to the wrapped kernel."""
        from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy

        kernel, _ = self._kernel()
        proxy = KernelLaunchMetaProxy(kernel, {"num_warps": 4})
        assert proxy.attribute == "kernel-attribute"

    def test_patch_kernel_launch_meta_installs_once(self, monkeypatch):
        """patch_kernel_launch_meta wraps the attribute exactly once."""
        from vllm_fl.patches.triton_kernel import (
            KernelLaunchMetaProxy,
            patch_kernel_launch_meta,
        )

        kernel, calls = self._kernel()
        module = SimpleNamespace(my_kernel=kernel)
        patch_kernel_launch_meta(module, "my_kernel", {"num_warps": 2})
        try:
            assert isinstance(module.my_kernel, KernelLaunchMetaProxy)
            module.my_kernel[(1,)]()
            assert calls[-1][1] == {"num_warps": 2}
        finally:
            module.my_kernel = kernel

    def test_patch_kernel_launch_meta_same_overrides_idempotent(self, monkeypatch):
        """Re-applying identical overrides is a no-op."""
        from vllm_fl.patches.triton_kernel import patch_kernel_launch_meta

        kernel, _ = self._kernel()
        module = SimpleNamespace(my_kernel=kernel)
        patch_kernel_launch_meta(module, "my_kernel", {"num_warps": 2})
        try:
            first = module.my_kernel
            patch_kernel_launch_meta(module, "my_kernel", {"num_warps": 2})
            assert module.my_kernel is first
        finally:
            module.my_kernel = kernel

    def test_patch_kernel_launch_meta_conflicting_overrides_raise(self, monkeypatch):
        """Re-applying different overrides raises RuntimeError."""
        from vllm_fl.patches.triton_kernel import patch_kernel_launch_meta

        kernel, _ = self._kernel()
        module = SimpleNamespace(my_kernel=kernel)
        patch_kernel_launch_meta(module, "my_kernel", {"num_warps": 2})
        try:
            with pytest.raises(RuntimeError, match="already has launch overrides"):
                patch_kernel_launch_meta(module, "my_kernel", {"num_warps": 8})
        finally:
            module.my_kernel = kernel


# ===========================================================================
# qsa_cache_hint
# ===========================================================================


class _FakeQsaKernel:
    """Kernel object mimicking a Triton JITFunction's source surface."""

    def __init__(self, src, device_caches=None):
        self.src = src
        self.device_caches = {} if device_caches is None else device_caches
        self.updates = []
        self.original_src = None

    def _unsafe_update_src(self, source):
        self.updates.append(source)
        self.src = source  # the real primitive replaces the cached source

    def _fl_original_src_marker(self):
        return self.original_src


_HINTED_SRC = (
    "def _qsa_mqa_paged_prefill_kernel(k_cache_ptr, other):\n"
    '    keys = tl.load(k_cache_ptr + other, eviction_policy="evict_first")\n'
)
_UNHINTED_SRC = (
    "def _qsa_mqa_paged_prefill_kernel(k_cache_ptr, other):\n"
    "    keys = tl.load(k_cache_ptr + other)\n"
)


@pytest.fixture
def qsa(monkeypatch):
    """Load the QSA patch module (pure stdlib imports at module scope)."""
    return _load(
        _ROOT / "vllm_fl/patches/qsa_cache_hint.py",
        "supp_qsa_cache_hint_under_test",
        monkeypatch,
    )


def qsa_kernel_names():
    """The two kernel names the patch targets."""
    return (
        "_qsa_mqa_paged_prefill_kernel",
        "_qsa_mqa_paged_uniform_kernel",
    )


class TestWithoutKCacheEvictionHint:
    """``_without_k_cache_eviction_hint`` AST surgery."""

    def test_removes_the_eviction_hint(self, qsa):
        """The eviction_policy keyword is removed; everything else survives."""
        result = qsa._without_k_cache_eviction_hint(_FakeQsaKernel(_HINTED_SRC))
        assert result is not None
        assert "eviction_policy" not in result
        # The load itself is intact, minus only the keyword.
        assert "keys = tl.load(k_cache_ptr + other)" in result
        assert "k_cache_ptr" in result and "other" in result

    def test_returns_none_when_already_patched(self, qsa):
        """A source without the hint reports None (already fixed)."""
        assert (
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(_UNHINTED_SRC)) is None
        )

    def test_rejects_missing_k_cache_load(self, qsa):
        """Zero matching K-cache loads raise the revalidation error."""
        src = "def _k(other):\n    keys = tl.load(other)\n"
        with pytest.raises(RuntimeError, match="K-cache load changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_multiple_k_cache_loads(self, qsa):
        """Two matching loads raise instead of guessing which to change."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            '    keys = tl.load(k_cache_ptr + 1, eviction_policy="evict_first")\n'
            '    keys = tl.load(k_cache_ptr + 2, eviction_policy="evict_first")\n'
        )
        with pytest.raises(RuntimeError, match="K-cache load changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_wrong_policy_value(self, qsa):
        """A different eviction policy value raises."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            '    keys = tl.load(k_cache_ptr + other, eviction_policy="evict_last")\n'
        )
        with pytest.raises(RuntimeError, match="eviction policy changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))

    def test_rejects_non_constant_policy(self, qsa):
        """A non-literal policy expression raises."""
        src = (
            "def _k(k_cache_ptr, other):\n"
            "    keys = tl.load(k_cache_ptr + other, eviction_policy=policy)\n"
        )
        with pytest.raises(RuntimeError, match="eviction policy changed"):
            qsa._without_k_cache_eviction_hint(_FakeQsaKernel(src))


class _QsaPatchEnv:
    """Install the full external boundary the patch function consults."""

    ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

    def __init__(self, monkeypatch, qsa):
        self.monkeypatch = monkeypatch
        self.kernels = {
            name: _FakeQsaKernel(_HINTED_SRC) for name in qsa_kernel_names()
        }
        self.qsa_indexer = ModuleType(
            "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
        )
        for name, kernel in self.kernels.items():
            setattr(self.qsa_indexer, name, kernel)

        self.triton_stub = ModuleType("triton")
        self.triton_stub.__version__ = "3.6.0"

        vllm_pkg = ModuleType("vllm")
        vllm_pkg.__path__ = []
        parent = ModuleType("vllm.models.qwen4_exp.nvidia.ops")
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
        monkeypatch.setitem(
            sys.modules, "vllm.models.qwen4_exp.nvidia.ops", parent
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
            self.qsa_indexer,
        )
        monkeypatch.setitem(sys.modules, "triton", self.triton_stub)

        self._original_capability = torch.cuda.get_device_capability
        self._original_version = qsa.version
        self.config = SimpleNamespace(architecture=self.ARCHITECTURE)
        self.device = SimpleNamespace(type="cuda")

    def set_triton_version(self, version: str):
        self.triton_stub.__version__ = version

    def set_capability(self, capability):
        torch.cuda.get_device_capability = (
            (lambda device=None: capability)
            if capability is not None
            else self._original_capability
        )

    def set_flagtree_version(self, resolver):
        """Point the module-level ``version`` at a custom resolver."""
        self.qsa_module.version = resolver

    @property
    def qsa_module(self):
        return sys.modules["supp_qsa_cache_hint_under_test"]

    def restore(self):
        torch.cuda.get_device_capability = self._original_capability
        module = sys.modules.get("supp_qsa_cache_hint_under_test")
        if module is not None:
            module.version = self._original_version


@pytest.fixture
def qsa_env(monkeypatch, qsa):
    """Install the patch's external boundaries with defaults that succeed."""
    env = _QsaPatchEnv(monkeypatch, qsa)
    env.set_flagtree_version(lambda name: "0.1.0")
    env.set_capability((9, 0))
    try:
        yield env
    finally:
        env.restore()


class TestPatchQsaCacheHintGates:
    """Every early-return gate of ``patch_qsa_cache_hint``."""

    def test_success_patches_both_kernels(self, qsa, qsa_env):
        """All gates open: both kernels lose their hint and keep a backup."""
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
        for kernel in qsa_env.kernels.values():
            assert len(kernel.updates) == 1
            assert "eviction_policy" not in kernel.updates[0]
            # The original source is preserved before mutation.
            assert getattr(kernel, "_fl_qsa_original_src", None) == _HINTED_SRC

    def test_second_call_reports_no_updates(self, qsa, qsa_env):
        """After patching, the changed sources yield no further updates."""
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False
        for kernel in qsa_env.kernels.values():
            assert len(kernel.updates) == 1

    def test_wrong_architecture_returns_false(self, qsa, qsa_env):
        """Non-Qwen4Exp architectures never reach the kernel imports."""
        config = SimpleNamespace(architecture="LlamaForCausalLM")
        assert qsa.patch_qsa_cache_hint(config, qsa_env.device) is False
        assert all(not k.updates for k in qsa_env.kernels.values())

    @pytest.mark.parametrize("triton_version", ["3.9.0", "3.5.1", "4.0.0"])
    def test_wrong_triton_version_returns_false(self, qsa, qsa_env, triton_version):
        """Only the 3.6 series proceeds to the flagtree check."""
        qsa_env.set_triton_version(triton_version)
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False
        assert all(not k.updates for k in qsa_env.kernels.values())

    def test_triton_36_patch_series_proceeds(self, qsa, qsa_env):
        """Any 3.6.x micro version passes the version gate."""
        for version in ("3.6.0", "3.6.2"):
            qsa_env.set_triton_version(version)
            assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is True
            # Reset for the next iteration of the parametrized style loop.
            for kernel in qsa_env.kernels.values():
                kernel.updates.clear()
                kernel.src = _HINTED_SRC

    def test_missing_flagtree_returns_false(self, qsa, qsa_env):
        """A machine without flagtree installed bails out early."""
        from importlib.metadata import PackageNotFoundError

        def resolver(name):
            if name == "flagtree":
                raise PackageNotFoundError(name)
            return "0"

        qsa_env.set_flagtree_version(resolver)
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False
        assert all(not k.updates for k in qsa_env.kernels.values())

    def test_non_cuda_device_returns_false(self, qsa, qsa_env):
        """A non-CUDA device never reaches the kernel imports."""
        device = SimpleNamespace(type="cpu")
        assert qsa.patch_qsa_cache_hint(qsa_env.config, device) is False

    def test_hip_build_returns_false(self, qsa, qsa_env, monkeypatch):
        """AMD HIP builds are excluded from the workaround."""
        monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False

    @pytest.mark.parametrize("capability", [(8, 0), (9, 1), (10, 0)])
    def test_non_hopper_capability_returns_false(self, qsa, qsa_env, capability):
        """Only Hopper (9, 0) triggers the patch."""
        qsa_env.set_capability(capability)
        assert qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device) is False

    def test_compiled_kernel_raises(self, qsa, qsa_env):
        """A kernel with live device caches refuses the source change."""
        first, second = qsa_kernel_names()
        qsa_env.kernels[first].device_caches = {"cuda:0": object()}
        with pytest.raises(RuntimeError, match="already compiled"):
            qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device)
        # The untouched second kernel was never mutated either: both are
        # validated before either is changed.
        assert qsa_env.kernels[second].updates == []

    def test_architectures_both_accepted(self, qsa, qsa_env):
        """Both Qwen4Exp architecture spellings pass the arch gate."""
        qsa.patch_qsa_cache_hint(qsa_env.config, qsa_env.device)
        for kernel in qsa_env.kernels.values():
            kernel.updates.clear()
            kernel.src = _HINTED_SRC
        causal = SimpleNamespace(architecture="Qwen4ExpForCausalLM")
        assert qsa.patch_qsa_cache_hint(causal, qsa_env.device) is True


# ===========================================================================
# arm_cpu_gdn
# ===========================================================================


@pytest.fixture
def arm_gdn(monkeypatch):
    """Load the ARM CPU GDN patch module with a plain logging logger."""
    logger = _FakeLogger()
    vllm_stub = ModuleType("vllm")
    vllm_stub.__path__ = []
    gdn_attn = ModuleType("vllm.v1.attention.backends.gdn_attn")

    class GDNAttentionMetadataBuilder:
        def build(self, *args, **kwargs):
            raise AssertionError("original build must not run")

    gdn_attn.GDNAttentionMetadataBuilder = GDNAttentionMetadataBuilder
    return (
        _load(
            _ROOT / "vllm_fl/patches/arm_cpu_gdn.py",
            "supp_arm_cpu_gdn_under_test",
            monkeypatch,
            **{
                "vllm": vllm_stub,
                "vllm.v1": ModuleType("vllm.v1"),
                "vllm.v1.attention": ModuleType("vllm.v1.attention"),
                "vllm.v1.attention.backends": ModuleType(
                    "vllm.v1.attention.backends"
                ),
                "vllm.v1.attention.backends.gdn_attn": gdn_attn,
            },
        ),
        GDNAttentionMetadataBuilder,
    )


class TestArmCpuGdn:
    def _metadata(self, state_indices):
        return SimpleNamespace(non_spec_state_indices_tensor=state_indices)

    def _patched_builder(self, module, metadata):
        """Install the patch on a local builder whose build returns metadata."""

        class Builder:
            def build(self, *args, **kwargs):
                return metadata

        module.apply_arm_cpu_gdn_state_indices_patch(Builder)
        return Builder

    def test_explicit_builder_class_used(self, arm_gdn):
        """A strided CPU state tensor is cloned into contiguous form."""
        module, _ = arm_gdn

        # A [:, 0] view: is_contiguous() reports True on some stacks, but the
        # last-dim stride is 2 — exactly the case the patch targets.
        strided_view = torch.arange(8).reshape(4, 2)[:, :1]
        metadata = self._metadata(strided_view)
        Builder = self._patched_builder(module, metadata)
        try:
            instance = object.__new__(Builder)
            result = Builder.build(instance)
            assert result is metadata
            assert result.non_spec_state_indices_tensor.stride(-1) == 1
            assert torch.equal(
                result.non_spec_state_indices_tensor.reshape(-1),
                strided_view.reshape(-1),
            )
        finally:
            delattr(Builder, "build")

    def test_patch_is_idempotent(self, arm_gdn):
        """A second install call reports False (marker present)."""
        module, _ = arm_gdn
        Builder = self._patched_builder(module, None)
        try:
            assert module.apply_arm_cpu_gdn_state_indices_patch(Builder) is False
        finally:
            delattr(Builder, "build")

    def test_cpu_stride_one_left_alone(self, arm_gdn):
        """A contiguous CPU state tensor passes through unchanged (same object)."""
        module, _ = arm_gdn

        contiguous = torch.arange(4).reshape(2, 2)
        metadata = self._metadata(contiguous)
        Builder = self._patched_builder(module, metadata)
        try:
            instance = object.__new__(Builder)
            out = Builder.build(instance)
            assert out.non_spec_state_indices_tensor is contiguous
        finally:
            delattr(Builder, "build")

    def test_none_state_indices_untouched(self, arm_gdn):
        """A metadata with None state indices returns unchanged."""
        module, _ = arm_gdn

        metadata = self._metadata(None)
        Builder = self._patched_builder(module, metadata)
        try:
            instance = object.__new__(Builder)
            out = Builder.build(instance)
            assert out is metadata
            assert out.non_spec_state_indices_tensor is None
        finally:
            delattr(Builder, "build")

    def test_cuda_state_indices_untouched(self, arm_gdn):
        """A non-CPU state tensor is never cloned."""

        class _FakeCudaTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda")

        fake = torch.arange(4).reshape(2, 2).as_subclass(_FakeCudaTensor)
        module, _ = arm_gdn
        metadata = self._metadata(fake)
        Builder = self._patched_builder(module, metadata)
        try:
            instance = object.__new__(Builder)
            out = Builder.build(instance)
            assert out.non_spec_state_indices_tensor is fake
        finally:
            delattr(Builder, "build")


# ===========================================================================
# minicpmo_audio
# ===========================================================================


@pytest.fixture
def minicpmo(monkeypatch):
    """Load the MiniCPM-O audio budget patch against a fake upstream module."""

    class _UpstreamInfo:
        """The upstream estimate: image/video only, no audio."""

        def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
            return {"image": 4 * seq_len, "video": 16 * seq_len}

        def get_max_audio_tokens(self):
            return 4096

    class MiniCPMOProcessingInfo(_UpstreamInfo):
        """The production class: inherits the estimate (not in its own dict)."""

    minicpmo_mod = ModuleType("vllm.model_executor.models.minicpmo")
    minicpmo_mod.MiniCPMOProcessingInfo = MiniCPMOProcessingInfo
    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    me_pkg = ModuleType("vllm.model_executor")
    me_pkg.__path__ = []
    models_pkg = ModuleType("vllm.model_executor.models")
    models_pkg.__path__ = []

    module = _load(
        _ROOT / "vllm_fl/patches/minicpmo_audio.py",
        "supp_minicpmo_audio_under_test",
        monkeypatch,
        **{
            "vllm": vllm_pkg,
            "vllm.model_executor": me_pkg,
            "vllm.model_executor.models": models_pkg,
            "vllm.model_executor.models.minicpmo": minicpmo_mod,
        },
    )
    return module, MiniCPMOProcessingInfo


class TestMinicpmoAudio:
    def test_patch_adds_audio_budget(self, minicpmo):
        """The patched estimate inherits the image/video values and adds audio."""
        module, info_cls = minicpmo
        try:
            assert module.patch_minicpmo_audio_budget() is True
            info = info_cls.__new__(info_cls)
            result = info.get_mm_max_tokens_per_item(10, {"image": 1, "video": 1})
            assert result["image"] == 40
            assert result["video"] == 160
            assert result["audio"] == 4096
        finally:
            if "get_mm_max_tokens_per_item" in info_cls.__dict__:
                delattr(info_cls, "get_mm_max_tokens_per_item")

    def test_patch_is_idempotent(self, minicpmo):
        """A second install call reports False (the patched name is present)."""
        module, info_cls = minicpmo
        try:
            assert module.patch_minicpmo_audio_budget() is True
            assert module.patch_minicpmo_audio_budget() is False
        finally:
            if "get_mm_max_tokens_per_item" in info_cls.__dict__:
                delattr(info_cls, "get_mm_max_tokens_per_item")

    def test_existing_override_blocks_patch(self, minicpmo):
        """An upstream override already in __dict__ wins over the patch."""
        module, info_cls = minicpmo

        def native_override(self, seq_len, mm_counts):
            return {"native": True}

        info_cls.get_mm_max_tokens_per_item = native_override
        try:
            assert module.patch_minicpmo_audio_budget() is False
        finally:
            delattr(info_cls, "get_mm_max_tokens_per_item")
