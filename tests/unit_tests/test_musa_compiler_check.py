# Copyright (c) 2026 BAAI. All rights reserved.
"""Driver-free regression checks for the MUSA compiler preflight."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "docker/musa/check_compiler.py"
spec = importlib.util.spec_from_file_location("musa_compiler_check", SCRIPT)
compiler_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler_check)


class TestMusaCompilerCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "triton/__init__.py"
        self.origin.parent.mkdir()
        self.origin.write_text('__version__ = "3.6.0"\n', encoding="utf-8")
        backend = self.root / "triton/backends/mthreads/compiler.py"
        backend.parent.mkdir(parents=True)
        backend.touch()
        self.provider = SimpleNamespace(
            version="0.6.1a2+mthreads3.6",
            locate_file=lambda path: self.root / path,
        )
        self.standalone = None

        def distribution(name):
            if name == "flagtree" and self.provider is not None:
                return self.provider
            if name == "triton" and self.standalone is not None:
                return self.standalone
            raise compiler_check.metadata.PackageNotFoundError(name)

        self.addCleanup(patch.stopall)
        patch.object(
            compiler_check.metadata, "distribution", side_effect=distribution
        ).start()
        self.import_module = patch.object(
            compiler_check.importlib,
            "import_module",
            return_value=SimpleNamespace(
                __file__=str(self.origin), __version__="3.6.0"
            ),
        ).start()
        patch.object(compiler_check, "print", create=True).start()

    def test_metadata_only_accepts_pinned_and_source_builds_without_importing(self):
        for version in ("0.6.1a2+mthreads3.6", "0.6.0+mthreadsb97f8214"):
            with self.subTest(version=version):
                self.provider.version = version
                compiler_check.check_compiler(metadata_only=True)
        self.import_module.assert_not_called()

    def test_rejects_missing_flagtree(self):
        self.provider = None
        with self.assertRaisesRegex(RuntimeError, "FlagTree is not installed"):
            compiler_check.check_compiler(metadata_only=True)

    def test_rejects_overlapping_standalone_triton(self):
        self.standalone = SimpleNamespace(version="3.6.0")
        with self.assertRaisesRegex(RuntimeError, "same triton namespace"):
            compiler_check.check_compiler(metadata_only=True)

    def test_rejects_wrong_backend(self):
        self.provider.version = "0.6.1a2+ascend3.5"
        with self.assertRaisesRegex(RuntimeError, "MThreads backend"):
            compiler_check.check_compiler(metadata_only=True)

    def test_rejects_legacy_triton_even_with_mthreads_label(self):
        self.origin.write_text('__version__ = "3.2.0"\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "expected 3.6.0"):
            compiler_check.check_compiler(metadata_only=True)

    def test_rejects_missing_mthreads_compiler(self):
        (self.root / "triton/backends/mthreads/compiler.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "compiler is missing"):
            compiler_check.check_compiler(metadata_only=True)

    def test_full_check_accepts_matching_import(self):
        compiler_check.check_compiler()
        self.import_module.assert_called_once_with("triton")

    def test_full_check_rejects_shadowed_triton(self):
        self.import_module.return_value.__file__ = str(self.root / "other/triton.py")
        with self.assertRaisesRegex(RuntimeError, "Active Triton comes from"):
            compiler_check.check_compiler()

    def test_full_check_rejects_stale_import(self):
        self.import_module.return_value.__version__ = "3.2.0"
        with self.assertRaisesRegex(RuntimeError, "Active Triton is 3.2.0"):
            compiler_check.check_compiler()


if __name__ == "__main__":
    unittest.main()
