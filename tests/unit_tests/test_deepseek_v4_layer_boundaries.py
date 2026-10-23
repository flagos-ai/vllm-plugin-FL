# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for layer ownership and local launch metadata bindings."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/dsv4/attention"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LayerBoundaryTests(unittest.TestCase):
    def setUp(self):
        proxy_module = load("ds_meta_proxy", ROOT / "vllm_fl/patches/triton_kernel.py")

        class JIT:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: (grid, args, kwargs)

        self.jit_type = JIT
        fake = types.ModuleType("triton")
        fake.runtime = types.SimpleNamespace(JITFunction=JIT)
        fake.autotune = lambda **options: lambda fn: types.SimpleNamespace(
            fn=fn, options=options
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "triton": fake,
                "vllm_fl.patches.triton_kernel": proxy_module,
            },
        )
        self.modules.start()
        self.launch = load("ds_local_launch", OPS / "launch.py")

    def tearDown(self):
        self.modules.stop()

    def test_proxy_is_local_and_overrides_metadata(self):
        jit = self.jit_type()
        original = types.SimpleNamespace(fn=types.SimpleNamespace(fn=jit))
        bound = self.launch.local_launch(original, {"BK": 16, "num_warps": 4})
        self.assertIs(original.fn.fn, jit)
        self.assertIsNot(bound, original)
        _, args, kwargs = bound[(2,)](123, BK=64, num_warps=8)
        self.assertEqual(args, (123,))
        self.assertEqual(kwargs, {"BK": 16, "num_warps": 4})

    def test_autotuner_is_not_mutated(self):
        jit = self.jit_type()
        original = types.SimpleNamespace(fn=jit, configs=["original"])
        bound = self.launch.local_autotune(original, configs=["accepted"], key=["M"])
        self.assertEqual(original.configs, ["original"])
        self.assertIs(bound.fn, jit)
        self.assertEqual(bound.options["configs"], ["accepted"])

    def test_bad_wrapper_is_rejected(self):
        bad = types.SimpleNamespace()
        bad.fn = bad
        with self.assertRaises(TypeError):
            self.launch.local_launch(bad, {"num_warps": 4})
        with self.assertRaises(TypeError):
            self.launch.local_launch(object(), {"num_warps": 4})

    def test_examples_contain_no_source_patch_installer(self):
        example = ROOT / "examples/deepseek_v4_metax"
        self.assertFalse(list(example.rglob("*.patch")))
        self.assertFalse((example / "prepare_dependencies.py").exists())

    def test_tp_kernels_are_owned_by_fl(self):
        expected = {
            "decode.py": "_partial_model1_kernel_halfd_qkfold_blockdiag",
            "prefill.py": "triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag",
            "staged.py": "pv_replay",
        }
        for file, name in expected.items():
            tree = ast.parse((OPS / file).read_text())
            self.assertIn(
                name, {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            )
        headpack = (ROOT / "vllm_fl/attention/deepseek_v4/headpack.py").read_text()
        self.assertIn("vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.attention import prefill", headpack)
        self.assertNotIn("flaggems_vllm.ops.flashmla_sparse", headpack)

    def test_tuning_has_no_mhc_or_compiler_policy(self):
        tuning = load("ds_attention_tuning_test", OPS / "tuning.py")
        from dataclasses import asdict

        values = asdict(tuning.KernelPolicy())
        self.assertTrue(all(k.startswith(("decode_", "prefill_")) for k in values))
        self.assertEqual(values["decode_splits"], 4)

    def test_model_layer_registration_is_deferred(self):
        patches = ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/__init__.py"
        tree = ast.parse(patches.read_text())

        def imports_gdn(node):
            return isinstance(node, ast.ImportFrom) and any(
                alias.name == "gdn_linear_attn" for alias in node.names
            )

        self.assertFalse(any(imports_gdn(node) for node in tree.body))
        hook = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "register_model_patches"
        )
        self.assertTrue(any(imports_gdn(node) for node in ast.walk(hook)))
        entry = ast.parse((ROOT / "vllm_fl/__init__.py").read_text())
        model_hook = next(
            node
            for node in entry.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_model"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_model_patches"
                for node in ast.walk(model_hook)
            )
        )

    def test_retired_attention_variants_are_absent(self):
        retired = {
            "decode.py": {"_partial_model1_kernel", "_partial_model1_kernel_halfd"},
            "prefill.py": {"triton_flash_mla_sparse_fwd_halfd_serial"},
        }
        for filename, names in retired.items():
            tree = ast.parse((OPS / filename).read_text())
            functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            self.assertFalse(functions & names)
        self.assertFalse((ROOT / "vllm_fl/ops/prefill_topk_native_op.py").exists())

    def test_unchanged_adapters_are_reused(self):
        folder = ROOT / "vllm_fl/dispatch/backends/flaggems/impl/deepseek_v4_metax"
        for name in (
            "activation",
            "normalization",
            "rotary",
            "router_gemm",
            "fused_moe",
            "deepseek_v4_ops",
        ):
            self.assertFalse((folder / (name + ".py")).exists())
        self.assertFalse((folder.parents[1] / "deepseek_v4_metax.py").exists())
        provider = (folder.parent / "deepseek_v4.py").read_text()
        self.assertIn("def operator_impls", provider)
        self.assertNotIn("torch.distributed", provider)

    def test_w4a8_vendor_registrations_are_absent(self):
        path = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/dsv4/mctlass_ops.py"
        tree = ast.parse(path.read_text())
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse(any("w4a8" in name for name in functions))
        self.assertIn("cutlass_moe_mm_w8a8_get_kernel_m", functions)

    def test_indexer_math_has_no_framework_or_collective_imports(self):
        folder = OPS.parent / "indexer"
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    self.assertFalse(module.startswith((
                        "torch.distributed", "vllm.distributed",
                        "vllm_fl.ops", "vllm_fl.attention", "vllm_fl.dsv4_config",
                    )), (path, module))
                self.assertFalse(
                    isinstance(node, ast.ImportFrom)
                    and node.module == "vllm_fl"
                    and any(a.name == "dsv4_config" for a in node.names),
                    path,
                )

    def test_framework_attention_contains_no_triton_kernels(self):
        folder = ROOT / "vllm_fl/attention/deepseek_v4"
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.assertFalse(any(
                        isinstance(d, ast.Attribute) and d.attr in {"jit", "autotune"}
                        for d in node.decorator_list
                    ), (path, node.name))
        for name in ("row_sharding.py", "query_sharding.py"):
            self.assertIn("dist.all_gather_into_tensor", (folder / name).read_text())
        row = (folder / "row_sharding.py").read_text()
        self.assertIn("indexer.row_score import", row)
        self.assertIn("indexer.deep_gemm import", row)

    def test_moved_packages_do_not_eagerly_import_providers(self):
        for folder in (
            ROOT / "vllm_fl/attention/deepseek_v4",
            ROOT / "vllm_fl/patches/deepseek_v4_metax",
            OPS.parent / "indexer",
        ):
            tree = ast.parse((folder / "__init__.py").read_text())
            self.assertTrue(all(isinstance(node, ast.Expr) for node in tree.body))

    def test_old_ops_paths_and_loose_file_fallback_are_absent(self):
        retired = {
            "batched_logits", "paged_indexer_logits", "tree_topk", "indexer_m84_score",
            "indexer_query_sharding", "prefill_indexer_rowshard", "dsv4_headpack",
            "dsv4_serial", "dsv4_validation",
        }
        for name in retired:
            self.assertFalse((ROOT / "vllm_fl/ops" / (name + ".py")).exists())
        for path in (ROOT / "vllm_fl").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, {"vllm_fl.ops." + n for n in retired})
                    self.assertNotEqual(node.module, "tree_topk")

    def test_validation_keeps_its_lazy_embedding_dependency(self):
        path = ROOT / "vllm_fl/patches/deepseek_v4_metax/validation.py"
        tree = ast.parse(path.read_text())
        target = "vllm_fl.ops.dsv4_embedding"
        self.assertFalse(any(
            isinstance(n, ast.ImportFrom) and n.module == target for n in tree.body
        ))
        self.assertTrue(any(
            isinstance(n, ast.ImportFrom) and n.module == target for n in ast.walk(tree)
        ))

    def test_all_python_sources_parse(self):
        for path in OPS.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
