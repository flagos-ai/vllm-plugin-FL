# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU regressions for model scope, provider selection and import boundaries."""

import ast
import importlib.util
import json
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PREFIX = "vllm_fl.ops.minimax_m3"
FG = "vllm_fl.dispatch.backends.flaggems.impl.minimax_m3"
VENDOR = "vllm_fl/dispatch/backends/vendor/metax/impl/minimax_m3"


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StructureTests(unittest.TestCase):
    def registration(self, enabled):
        package = types.ModuleType(PREFIX)
        package.__path__ = []
        torch_ops = types.ModuleType(PREFIX + ".torch_ops")
        moe = types.ModuleType(FG + ".moe")
        names = (
            "gemma",
            "swiglu",
            "qknorm_rope_insert",
            "patch_embed",
            "vision_attention",
            "index_score",
            "index_topk",
            "index_decode",
            "sparse_attn",
            "sparse_decode",
        )
        for name in names:
            setattr(torch_ops, name, object())
        for name in (
            "moe_align_block_size",
            "invoke_fused_moe_triton_kernel",
            "moe_sum",
            "grouped_topk",
        ):
            setattr(moe, name, object())
        package.torch_ops = torch_ops
        library = types.ModuleType(FG)
        library.moe = moe
        t = load_file("_m3_structure_dispatch_types", "vllm_fl/dispatch/types.py")
        dispatch = types.ModuleType("vllm_fl.dispatch")
        for name in ("BackendImplKind", "BackendPriority", "OpImpl"):
            setattr(dispatch, name, getattr(t, name))
        modules = {
            PREFIX: package,
            PREFIX + ".torch_ops": torch_ops,
            FG: library,
            FG + ".moe": moe,
            "vllm_fl.dispatch": dispatch,
            "vllm_fl.minimax_m3_config": types.SimpleNamespace(
                is_enabled=lambda: enabled
            ),
        }
        result = []
        with patch.dict(sys.modules, modules):
            m = load_file(
                PREFIX + ".registration", "vllm_fl/ops/minimax_m3/registration.py"
            )
            m.vllm_fl_register(types.SimpleNamespace(register_impl=result.append))
        return result, t, torch_ops, moe

    def test_other_models_register_nothing(self):
        rows, _, _, _ = self.registration(False)
        self.assertEqual(rows, [])

    def test_policy_selects_exact_provider_and_function(self):
        rows, t, ops, moe = self.registration(True)
        self.assertEqual(len(rows), 14)
        self.assertEqual(len({r.op_name for r in rows}), 14)
        cfg = json.loads((ROOT / "examples/minimax_m3_metax/config.json").read_text())
        msa = {
            "index_score",
            "index_topk",
            "index_decode",
            "sparse_attn",
            "sparse_decode",
        }
        for row in rows:
            name = row.op_name.removeprefix("m3_")
            provider = ops if hasattr(ops, name) else moe
            self.assertIs(row.fn, getattr(provider, name))
            if name in msa:
                self.assertEqual(row.kind, t.BackendImplKind.VENDOR)
                self.assertEqual(row.impl_id, "vendor.metax.m3." + name)
                self.assertEqual(cfg["op_backends"][row.op_name], ["vendor:metax"])
                self.assertTrue(t.match_token(row, "vendor:metax"))
                self.assertFalse(t.match_token(row, "flagos"))
            else:
                self.assertEqual(row.kind, t.BackendImplKind.DEFAULT)
                self.assertIsNone(row.vendor)

    def test_interfaces_and_orchestration_do_not_define_jit_kernels(self):
        for folder in (
            "vllm_fl/models/minimax_m3",
            "vllm_fl/ops/minimax_m3",
            "vllm_fl/attention/minimax_m3",
        ):
            for p in (ROOT / folder).glob("*.py"):
                for node in ast.walk(ast.parse(p.read_text())):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertNotIn(
                            "triton.jit",
                            [ast.unparse(d) for d in node.decorator_list],
                            str(p),
                        )

    def test_local_calculation_has_no_reverse_model_or_ops_imports(self):
        for p in (ROOT / VENDOR).glob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(
                        module.startswith(
                            (
                                "vllm_fl.models.",
                                "vllm_fl.ops.minimax_m3",
                                "vllm_fl.attention.",
                            )
                        ),
                        str(p),
                    )


if __name__ == "__main__":
    unittest.main()
