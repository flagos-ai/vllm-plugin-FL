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
        library = types.ModuleType(FG)
        native_name = "vllm_fl.dispatch.backends.vendor.metax.impl.minimax_m3"
        native = types.ModuleType(native_name)
        groups = {
            "numerics": (
                "gemma",
                "swiglu",
                "qknorm_rope_insert",
                "router_bf16_load_fp32_dot",
            ),
            "vision": ("_patch_embed", "vision_attention"),
            "moe": (
                "moe_align_block_size",
                "invoke_fused_moe_triton_kernel",
                "moe_sum",
                "grouped_topk",
            ),
        }
        modules = {FG: library, native_name: native}
        providers = {}
        for group, names in groups.items():
            module = types.ModuleType(FG + "." + group)
            for name in names:
                setattr(module, name, object())
            setattr(library, group, module)
            modules[module.__name__] = module
            providers[group] = module
        for group, names in {
            "attention_launch": ("index_score", "prefill", "decode"),
            "moe_gemv": ("expert_gemv",),
            "sampling": ("stable_sort_keys",),
            "native_index": (
                "minimax_m3_index_score",
                "minimax_m3_index_topk",
                "minimax_m3_index_decode",
            ),
            "native_sparse": (
                "minimax_m3_sparse_attn",
                "minimax_m3_sparse_attn_decode",
            ),
        }.items():
            module = types.ModuleType(native_name + "." + group)
            for name in names:
                setattr(module, name, object())
            setattr(native, group, module)
            modules[module.__name__] = module
            providers[group] = module
        t = load_file("_m3_structure_dispatch_types", "vllm_fl/dispatch/types.py")
        dispatch = types.ModuleType("vllm_fl.dispatch")
        for name in ("BackendImplKind", "BackendPriority", "OpImpl"):
            setattr(dispatch, name, getattr(t, name))
        modules.update(
            {
                "vllm_fl.dispatch": dispatch,
                "vllm_fl.minimax_m3_config": types.SimpleNamespace(
                    is_enabled=lambda: enabled
                ),
            }
        )
        result = []
        with patch.dict(sys.modules, modules):
            module = load_file(
                PREFIX + ".registration", "vllm_fl/ops/minimax_m3/registration.py"
            )
            module.vllm_fl_register(types.SimpleNamespace(register_impl=result.append))
        return result, t, providers

    def test_other_models_register_nothing(self):
        rows, _, _ = self.registration(False)
        self.assertEqual(rows, [])

    def test_policy_selects_numerical_provider_and_preserves_native_fallback(self):
        rows, t, providers = self.registration(True)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r.op_name for r in rows}), 20)
        by_name = {r.op_name: r for r in rows}
        self.assertIs(
            by_name["m3_index_score"].fn, providers["attention_launch"].index_score
        )
        self.assertIs(
            by_name["m3_sparse_attn"].fn, providers["attention_launch"].prefill
        )
        self.assertIs(
            by_name["m3_sparse_decode"].fn, providers["attention_launch"].decode
        )
        self.assertIs(
            by_name["m3_router_bf16_fp32"].fn,
            providers["numerics"].router_bf16_load_fp32_dot,
        )
        self.assertIs(by_name["m3_expert_gemv"].fn, providers["moe_gemv"].expert_gemv)
        self.assertIs(
            by_name["m3_index_topk"].fn, providers["native_index"].minimax_m3_index_topk
        )
        self.assertNotIn("m3_index_decode", by_name)
        native = {
            "sampling_stable_sort",
            "index_score",
            "sparse_attn",
            "sparse_decode",
            "expert_gemv",
            "index_score_native",
            "index_topk",
            "index_decode_native",
            "sparse_attn_native",
            "sparse_decode_native",
        }
        cfg = json.loads((ROOT / "examples/minimax_m3_metax/config.json").read_text())
        for row in rows:
            name = row.op_name.removeprefix("m3_")
            if name in native:
                self.assertEqual(row.kind, t.BackendImplKind.VENDOR)
                self.assertEqual(row.vendor, "metax")
                self.assertEqual(cfg["op_backends"][row.op_name], ["vendor:metax"])
            else:
                self.assertEqual(row.kind, t.BackendImplKind.DEFAULT)
                self.assertIsNone(row.vendor)
                self.assertTrue(t.match_token(row, "flagos"))

    def test_interfaces_and_orchestration_do_not_define_jit_kernels(self):
        for folder in (
            "vllm_fl/models/minimax_m3_metax",
            "vllm_fl/ops/minimax_m3",
            "vllm_fl/attention/minimax_m3",
        ):
            for p in (ROOT / folder).glob("*.py"):
                for node in ast.walk(ast.parse(p.read_text())):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertNotIn(
                            "triton.jit",
                            [
                                ast.unparse(d).split("(", 1)[0]
                                for d in node.decorator_list
                            ],
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
