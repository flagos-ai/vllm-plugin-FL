# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.patches.fla_module_relocation``.

The module aliases the pre-0.25 ``vllm.model_executor.layers.fla`` prefix onto
``vllm.third_party.flash_linear_attention`` in ``sys.modules``.  Only the
external boundary (a fake ``vllm`` package on disk / in ``sys.modules``) is
stubbed; the relocation logic itself runs unmodified.

Semantics under test:

* install: full alias map (package + subpackage + children), parent
  attribute binding, broken-child tolerance, idempotency;
* no-op when the genuine old package is importable (vLLM <= 0.24);
* no-op when the new package is absent;
* ``_module_exists`` never raises on broken parent packages.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

import vllm_fl.patches.fla_module_relocation as rel

_OLD = rel._OLD_PREFIX
_NEW = rel._NEW_PREFIX


class _FakeVllmTree:
    """Materialize a fake ``vllm`` package on disk and import it.

    Writing real files (instead of bare ``sys.modules`` entries) exercises
    the plugin's ``find_spec`` / ``import_module`` probes the same way a
    genuine vllm install would.
    """

    def __init__(self, root: Path, with_old_fla=False, with_new_fla=True):
        self.root = root
        vllm_dir = root / "vllm"
        vllm_dir.mkdir(parents=True)
        (vllm_dir / "__init__.py").write_text("")
        me = vllm_dir / "model_executor"
        me.mkdir()
        (me / "__init__.py").write_text("")
        layers = me / "layers"
        layers.mkdir()
        (layers / "__init__.py").write_text("")
        if with_old_fla:
            old = layers / "fla"
            old.mkdir()
            (old / "__init__.py").write_text("MARKER = 'genuine_old_fla'\n")
        if with_new_fla:
            third_party = vllm_dir / "third_party"
            third_party.mkdir()
            (third_party / "__init__.py").write_text("")
            new = third_party / "flash_linear_attention"
            new.mkdir()
            (new / "__init__.py").write_text("MARKER = 'new_fla_pkg'\n")
            ops = new / "ops"
            ops.mkdir()
            (ops / "__init__.py").write_text("MARKER = 'new_fla_ops'\n")
            (ops / "chunk.py").write_text("MARKER = 'new_fla_chunk'\n")
            (ops / "broken.py").write_text("raise ImportError('broken child')\n")

    def install(self):
        """Put the fake tree first on sys.path.

        Any ``vllm*`` entries already in ``sys.modules`` (leaked stubs from
        earlier tests that fake the vllm boundary) are dropped first: their
        ``find_spec`` resolution would otherwise shadow the on-disk tree and
        make ``_module_exists`` report the whole package as absent.
        """
        self._saved_path = list(sys.path)
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in list(sys.modules)
            if name == "vllm" or name.startswith("vllm.")
        }
        for name in self._saved_modules:
            del sys.modules[name]
        sys.path.insert(0, str(self.root))

    def restore(self):
        """Remove the tree from sys.path and drop its cached modules."""
        sys.path[:] = self._saved_path
        for name in [
            name
            for name in list(sys.modules)
            if name == "vllm" or name.startswith("vllm.")
        ]:
            del sys.modules[name]
        for name, module in self._saved_modules.items():
            if module is not None:
                sys.modules[name] = module


@pytest.fixture
def fl_sys_path(fl_sys_modules, tmp_path):
    """Yield a builder for fake on-disk vllm trees, cleaned up after."""
    trees = []

    def _make(with_old_fla=False, with_new_fla=True):
        tree = _FakeVllmTree(
            tmp_path / f"tree_{len(trees)}",
            with_old_fla=with_old_fla,
            with_new_fla=with_new_fla,
        )
        tree.install()
        trees.append(tree)
        return tree

    yield _make
    for tree in reversed(trees):
        tree.restore()


class TestModuleExists:
    """``_module_exists`` must never raise, only report False."""

    def test_absent_module_reports_false(self):
        """A never-installed dotted name reports False."""
        assert rel._module_exists("definitely.not.installed.module") is False

    def test_broken_parent_raises_are_swallowed(self):
        """A parent module lacking __path__ raises, which reads as False."""
        parent = ModuleType("broken_parent_pkg")
        sys.modules["broken_parent_pkg"] = parent
        try:
            # find_spec raises ModuleNotFoundError for a non-package parent;
            # the helper converts that to False.
            assert rel._module_exists("broken_parent_pkg.child") is False
        finally:
            del sys.modules["broken_parent_pkg"]

    def test_module_without_spec_entry_raises_value_error(self):
        """A sys.modules entry with __spec__=None is treated as absent."""
        stub = ModuleType("nospec_pkg.sub")
        sys.modules["nospec_pkg.sub"] = stub
        try:
            assert rel._module_exists("nospec_pkg.sub") is False
        finally:
            del sys.modules["nospec_pkg.sub"]


class TestInstallApplies:
    """The successful alias installation path."""

    def test_installs_alias_for_package_and_children(self, fl_sys_path):
        """install() aliases the new package under the old prefix."""
        fl_sys_path()
        assert rel.install() is True
        new_pkg = importlib.import_module(_NEW)
        assert sys.modules[_OLD] is new_pkg
        assert sys.modules[_OLD + ".ops"] is sys.modules[_NEW + ".ops"]
        assert sys.modules[_OLD + ".ops.chunk"] is sys.modules[_NEW + ".ops.chunk"]
        # Aliased children resolve to the real new modules with content.
        assert sys.modules[_OLD + ".ops.chunk"].MARKER == "new_fla_chunk"

    def test_binds_fla_attribute_on_layers_package(self, fl_sys_path):
        """The parent layers package gains a ``fla`` attribute."""
        fl_sys_path()
        rel.install()
        layers = importlib.import_module("vllm.model_executor.layers")
        assert layers.fla is sys.modules[_NEW]
        # The old-prefix dotted import works through that attribute.
        from vllm.model_executor.layers.fla.ops import chunk

        assert chunk.MARKER == "new_fla_chunk"

    def test_broken_child_is_skipped_not_fatal(self, fl_sys_path):
        """A submodule failing to import is skipped without aborting."""
        fl_sys_path()
        rel.install()
        assert (_OLD + ".ops.broken") not in sys.modules
        assert (_OLD + ".ops.chunk") in sys.modules

    def test_install_is_idempotent(self, fl_sys_path):
        """A second install() leaves the same aliases in place."""
        fl_sys_path()
        assert rel.install() is True
        first = sys.modules[_OLD]
        first_ops = sys.modules[_OLD + ".ops"]
        # Fresh state would re-add the same objects; second call must not
        # swap them for new module instances.
        rel.install()
        assert sys.modules[_OLD] is first
        assert sys.modules[_OLD + ".ops"] is first_ops


class TestInstallNoOps:
    """The two deliberate no-op conditions."""

    def test_noop_when_genuine_old_package_exists(self, fl_sys_path):
        """vLLM <= 0.24 ships the real fla package: leave it alone."""
        fl_sys_path(with_old_fla=True)
        assert rel.install() is False
        assert _OLD not in sys.modules
        genuine = importlib.import_module(_OLD)
        assert genuine.MARKER == "genuine_old_fla"

    def test_noop_when_new_package_absent(self, fl_sys_path):
        """Without the relocated package there is nothing to alias."""
        fl_sys_path(with_new_fla=False)
        assert rel.install() is False
        assert _OLD not in sys.modules
        assert rel._module_exists(_NEW) is False

    def test_alias_respects_prefill_of_old_prefix(self, fl_sys_path):
        """Pre-existing old-prefix sys.modules entries are preserved."""
        fl_sys_path()
        preexisting = ModuleType(_OLD)
        sys.modules[_OLD] = preexisting
        try:
            assert rel.install() is True
            # The package-level alias was NOT overwritten.
            assert sys.modules[_OLD] is preexisting
            # But children still get aliased.
            assert sys.modules[_OLD + ".ops"] is sys.modules[_NEW + ".ops"]
        finally:
            del sys.modules[_OLD]


class TestPublicAlias:
    """The module's public re-export name."""

    def test_apply_alias_is_install(self):
        """apply_fla_module_relocation is the documented public entry."""
        assert rel.apply_fla_module_relocation is rel.install
