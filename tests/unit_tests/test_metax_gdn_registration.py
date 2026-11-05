# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle tests; dependency doubles do not validate device execution."""

import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATCHES = "vllm_fl.dispatch.backends.vendor.metax.patches"
GDN = PATCHES + ".gdn_linear_attn"
NATIVE = "vllm.model_executor.layers.mamba.gdn_linear_attn"


def _registration_statement():
    tree = ast.parse((ROOT / "vllm_fl/worker/model_runner.py").read_text())
    runner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    init = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    statements = [
        n
        for n in init.body
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module == PATCHES
            and any(alias.name == "gdn_linear_attn" for alias in child.names)
            for child in ast.walk(n)
        )
    ]
    if len(statements) != 1:
        raise AssertionError("GDN must be registered in ModelRunnerFL initialization")
    sampler = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "sampler" for t in n.targets)
    )
    if statements[0].lineno >= sampler.lineno:
        raise AssertionError(
            "Register vendor layers before constructing runner consumers"
        )
    return compile(
        ast.Module(body=statements, type_ignores=[]),
        str(ROOT / "vllm_fl/worker/model_runner.py"),
        "exec",
    )


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class TestMetaXGDNRegistration(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in list(sys.modules):
            if (
                name == "vllm_fl"
                or name.startswith("vllm_fl.")
                or name == "vllm"
                or name.startswith("vllm.")
            ):
                del sys.modules[name]
        # Import the real patch package without initializing unrelated backends.
        parts = PATCHES.split(".")
        for i in range(1, len(parts)):
            name = ".".join(parts[:i])
            _module(name, __path__=[str(ROOT.joinpath(*parts[:i]))])
        package = ROOT.joinpath(*parts, "__init__.py")
        for node in ast.parse(package.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                for alias in node.names:
                    if alias.name != "gdn_linear_attn":
                        _module(PATCHES + "." + alias.name)
        for name in [
            "vllm",
            "vllm.utils",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.mamba",
            "vllm.transformers_utils",
            "vllm.transformers_utils.configs",
        ]:
            _module(name, __path__=[])
        _module("vllm.utils.import_utils", has_triton_kernels=lambda: True)
        # This module deliberately lacks its class while vLLM is initializing.
        self.native = _module(NATIVE)
        _module("torch", Tensor=object)
        _module("einops", rearrange=lambda value, *_a, **_kw: value)
        _module("vllm.config", VllmConfig=object)
        _module("vllm.transformers_utils.configs.qwen3_next", Qwen3NextConfig=object)
        self.registrations = []

    def _finish_vllm_imports(self):
        registrations = self.registrations

        class NativeGDN:
            def __init__(self, **kwargs):
                self.prefix = kwargs.get("prefix", "")

            @classmethod
            def register_oot(cls, implementation):
                registrations.append(implementation)
                return implementation

        self.native.GatedDeltaNetAttention = NativeGDN

    def _initialize_registration(self, platform):
        # Execute the production initialization statement, not a copied gate.
        exec(_registration_statement(), {"current_platform": platform})

    def test_patch_package_does_not_import_partially_initialized_gdn(self):
        importlib.import_module(PATCHES)
        self.assertNotIn(GDN, sys.modules)
        self.assertEqual(self.registrations, [])

    def test_metax_registers_before_layer_construction(self):
        importlib.import_module(PATCHES)
        self._finish_vllm_imports()
        self._initialize_registration(types.SimpleNamespace(vendor_name="metax"))
        self.assertEqual(len(self.registrations), 1)
        layer = self.registrations[0](
            config=None, vllm_config=None, prefix="model.layers.0"
        )
        self.assertEqual(type(layer).__name__, "MacaGatedDeltaNetAttention")
        self.assertEqual(layer.prefix, "model.layers.0")
        self.assertTrue(layer.create_in_proj_qkvz)

    def test_repeated_initialization_reuses_module_registration(self):
        importlib.import_module(PATCHES)
        self._finish_vllm_imports()
        platform = types.SimpleNamespace(vendor_name="metax")
        self._initialize_registration(platform)
        original = self.registrations[0]
        self._initialize_registration(platform)
        self.assertEqual(self.registrations, [original])

    def test_other_platforms_do_not_import_metax_gdn(self):
        importlib.import_module(PATCHES)
        for platform in [
            types.SimpleNamespace(vendor_name="nvidia"),
            types.SimpleNamespace(vendor_name="ascend"),
            types.SimpleNamespace(),
        ]:
            with self.subTest(platform=platform):
                self._initialize_registration(platform)
                self.assertNotIn(GDN, sys.modules)
                self.assertEqual(self.registrations, [])

    def test_late_import_failure_is_visible_and_retry_is_not_cached(self):
        importlib.import_module(PATCHES)
        platform = types.SimpleNamespace(vendor_name="metax")
        with self.assertRaises(ImportError):
            self._initialize_registration(platform)
        self.assertNotIn(GDN, sys.modules)
        self._finish_vllm_imports()
        self._initialize_registration(platform)
        self.assertEqual(len(self.registrations), 1)


if __name__ == "__main__":
    unittest.main()
