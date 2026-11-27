# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise MetaX import hooks without requiring the vendor Triton runtime."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SHIMS = (
    Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/vendor/metax/shims"
)

# A fresh interpreter keeps these synthetic Triton modules and import hooks
# away from the real runtime used by the rest of the test suite. Loading the
# shim files as a package also catches stale imports of their old top-level
# site-packages names.
_DELAYED_IMPORT = """
import ast
import importlib
import importlib.abc
import importlib.util
import os
import sys
import types

vendor = types.ModuleType("metax_boot_test")
vendor.__path__ = [os.path.dirname(sys.argv[1])]
sys.modules[vendor.__name__] = vendor
package_name = vendor.__name__ + ".shims"
package = types.ModuleType(package_name)
package.__path__ = [sys.argv[1]]
sys.modules[package_name] = package

for name in ("triton", "triton.compiler"):
    parent = types.ModuleType(name)
    parent.__path__ = []
    parent.__version__ = os.environ["METAX_TEST_TRITON_VERSION"]
    sys.modules[name] = parent

target = "triton.compiler.code_generator"

class CodeGenerator:
    def visit_Assign(self, node):
        return node.targets[0].id

    def visit_AnnAssign(self, node):
        if node.annotation.id == "constexpr":
            return "constexpr"
        return self.visit_Assign(node)

    def visit_BoolOp(self, node):
        if len(node.values) != 2:
            raise ValueError("chained boolean operators are not supported")
        return tuple(
            self.visit_BoolOp(value) if isinstance(value, ast.BoolOp) else value.id
            for value in node.values
        )

class Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.CodeGenerator = CodeGenerator

class SyntheticTritonFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton.compiler.code_generator":
            return importlib.util.spec_from_loader(fullname, Loader())
        return None

sys.meta_path.insert(0, SyntheticTritonFinder())
annassign = importlib.import_module(package_name + "._fl_metax_annassign_compat")
boolchain = importlib.import_module(package_name + "._fl_metax_boolchain_compat")
assert target not in sys.modules
assert "_fl_metax_annassign_compat" not in sys.modules

module = importlib.import_module(target)
generator = module.CodeGenerator()
if os.environ["METAX_TEST_TRITON_VERSION"].startswith("3.6"):
    assert not annassign.status()["patched"]
    assert not boolchain.status()["patched"]
    assert not boolchain.status()["enabled"]
    sys.exit(0)
assert boolchain.status()["patched"]
assert generator.visit_BoolOp(ast.parse("a or b or c", mode="eval").body) == (("a", "b"), "c")
annotation = ast.parse("x: int = 1").body[0]
if os.environ["FL_METAX_ANNASSIGN_PATCH"] == "0":
    assert not annassign.status()["patched"]
    try:
        generator.visit_AnnAssign(annotation)
    except AttributeError:
        pass
    else:
        raise AssertionError("disabled AnnAssign shim was applied")
else:
    assert annassign.status()["patched"]
    assert generator.visit_AnnAssign(annotation) == "x"
    assert generator.visit_AnnAssign(ast.parse("x: constexpr = 1").body[0]) == "constexpr"
    # Reinstalling the sibling must not wrap visit_Assign again.
    original = module.CodeGenerator.visit_Assign
    boolchain.install()
    assert module.CodeGenerator.visit_Assign is original
"""


class TestMetaXBootShims(unittest.TestCase):
    def test_delayed_triton_import_applies_both_ast_patches(self):
        self._run_delayed_import(annassign_enabled=True)

    def test_delayed_triton_import_preserves_sibling_kill_switch(self):
        self._run_delayed_import(annassign_enabled=False)

    def test_modern_triton_compiler_is_not_patched(self):
        self._run_delayed_import(
            annassign_enabled=True, triton_version="3.6.0+metax3.9.0.8"
        )

    def _run_delayed_import(
        self, *, annassign_enabled, triton_version="3.0.0+metax3.7.0.7"
    ):
        env = dict(os.environ)
        env["METAX_TEST_TRITON_VERSION"] = triton_version
        env["FL_METAX_BOOLCHAIN_PATCH"] = "1"
        env["FL_METAX_ANNASSIGN_PATCH"] = "1" if annassign_enabled else "0"
        result = subprocess.run(
            [sys.executable, "-c", _DELAYED_IMPORT, str(_SHIMS)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
