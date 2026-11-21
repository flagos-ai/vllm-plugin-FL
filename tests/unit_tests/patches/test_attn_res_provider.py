# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing contract tests. Run this file directly with Python.

These mocks do not qualify the Triton kernel. GPU numerical, graph and compile
qualification lives in tests/functional_tests/test_attn_res_pr737.py.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def load_patch():
    spec = importlib.util.spec_from_file_location(
        "attn_res_provider_contract", ROOT / "vllm_fl/patches/attn_res_provider.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoutingContract(unittest.TestCase):
    def setUp(self):
        self.module = load_patch()
        self.config = SimpleNamespace(
            architecture="KimiK3ForConditionalGeneration",
            dtype="bf16",
            hf_text_config=SimpleNamespace(attn_res_block_size=6, num_hidden_layers=48),
        )
        self.device = SimpleNamespace(type="cuda")
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(get_device_capability=lambda _: (9, 0)),
        )
        self.env = patch.dict(os.environ, VLLM_FL_ATTN_RES_PROVIDER="1")
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_patch(self):
        with patch.dict(sys.modules, torch=self.torch):
            return self.module.patch_attn_res_provider(self.config, self.device)

    def test_disabled_does_not_load_torch_or_provider(self):
        with (
            patch.dict(os.environ, VLLM_FL_ATTN_RES_PROVIDER="0"),
            patch.object(
                self.module, "_load_provider", side_effect=AssertionError("loaded")
            ),
        ):
            self.assertFalse(self.run_patch())

    def test_env_typo_fails(self):
        with (
            patch.dict(os.environ, VLLM_FL_ATTN_RES_PROVIDER="auto"),
            self.assertRaises(ValueError),
        ):
            self.run_patch()

    def test_other_model_does_not_load_provider(self):
        self.config.architecture = "Qwen3ForCausalLM"
        with patch.object(self.module, "_load_provider", side_effect=AssertionError):
            self.assertFalse(self.run_patch())

    def test_strict_hardware_dtype_block_gates(self):
        for attribute, value in [("dtype", "fp16"), ("dtype", "fp8")]:
            with self.subTest(value=value), patch.object(self.config, attribute, value):
                self.assertFalse(self.run_patch())
        self.torch.cuda.get_device_capability = lambda _: (10, 0)
        self.assertFalse(self.run_patch())
        self.torch.cuda.get_device_capability = lambda _: (9, 0)
        self.config.hf_text_config.num_hidden_layers = 42
        self.assertFalse(self.run_patch())
        self.config.hf_text_config.attn_res_block_size = None
        self.assertFalse(self.run_patch())

    def test_updates_model_import_binding_ops_export_and_warmup(self):
        original = Mock(name="native")
        original._fl_attn_res_pr737 = False
        native = SimpleNamespace(attn_res=original)
        ops = SimpleNamespace(attn_res=original)
        model = SimpleNamespace(attn_res=original)
        modules = {
            f"{self.module._BASE}.ops.attn_res": native,
            f"{self.module._BASE}.ops": ops,
            f"{self.module._BASE}.model": model,
        }
        provider = Mock(return_value="provider_output")
        with (
            patch.object(self.module, "_load_provider", return_value=provider),
            patch.object(self.module, "_create_dispatch", return_value=provider),
            patch.object(
                self.module.importlib, "import_module", side_effect=modules.__getitem__
            ),
        ):
            self.assertTrue(self.run_patch())
            self.assertIs(model.attn_res, ops.attn_res)
            self.assertIs(model.attn_res, native.attn_res)
            self.assertIs(model.attn_res._fl_native_attn_res, original)
            args = ("prefix", "delta", "blocks", "nw", "qw", "ow", 3, 3, 1e-5, 1e-5)
            self.assertEqual(model.attn_res(*args), "provider_output")
            provider.assert_called_once_with(*args)
            original.assert_not_called()
            self.assertFalse(self.run_patch())
            profiles = native.get_attn_res_triton_warmup_profiles(8)
            self.assertIn((0, False, 0, True), profiles)
            self.assertIn((8, True, -1, False), profiles)
            self.assertIn((0, False, -1, True), profiles)

    def test_applicable_dependency_failure_propagates(self):
        with (
            patch.object(
                self.module, "_load_provider", side_effect=ImportError("provider")
            ),
            self.assertRaisesRegex(ImportError, "provider"),
        ):
            self.run_patch()

    def test_custom_op_records_both_mutations_and_executes_only_provider(self):
        class CustomOp:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, *args):
                return self.fn(*args)

            def register_fake(self, fn):
                self.fake = fn
                return fn

        raw = Mock(return_value="result")
        register = Mock(side_effect=lambda name, fn, **kwargs: CustomOp(fn))
        torch = SimpleNamespace(
            library=SimpleNamespace(custom_op=register),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: True),
        )
        dispatch = self.module._create_dispatch(raw, torch)
        kwargs = register.call_args.kwargs
        self.assertEqual(kwargs["mutates_args"], ("prefix", "blocks"))
        self.assertIn("Tensor(a!) prefix", kwargs["schema"])
        self.assertIn("Tensor(b!) blocks", kwargs["schema"])
        prefix = SimpleNamespace(shape=(64, 7168))
        args = (prefix, None, "blocks", "nw", "qw", "ow", 3, -1, 1e-5, 1e-5)
        self.assertEqual(dispatch(*args), "result")
        raw.assert_called_once_with(*args)
        self.assertEqual(self.module.attn_res_provider_hits(), {"post:capture": 1})
        raw.side_effect = RuntimeError("compile failed")
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            dispatch(*args)
        self.assertEqual(self.module.attn_res_provider_hits(), {"post:capture": 1})


class KernelPlans(unittest.TestCase):
    def test_all_pr_config_pruning_paths_nonempty_and_mutating_launch_is_fixed(self):
        # Import decorators/configs with inert CPU shims, then exercise only
        # pure launch-plan selection. This does not execute a Triton kernel.
        identity = lambda *args, **kwargs: lambda fn: fn

        class Config:
            def __init__(self, kwargs, num_warps, num_stages):
                self.kwargs, self.num_warps, self.num_stages = (
                    kwargs,
                    num_warps,
                    num_stages,
                )

        triton = ModuleType("triton")
        triton.jit = identity
        triton.Config = Config
        language = ModuleType("triton.language")
        language.constexpr = object()
        triton.language = language
        package = ModuleType("attn_pr737_contract")
        package.__path__ = [str(ROOT / "vllm_fl/third_party/attn_res_pr737")]
        utils = SimpleNamespace(
            has_triton_tle=lambda *a: False, libentry=identity, libtuner=identity
        )
        shims = {
            "torch": ModuleType("torch"),
            "triton": triton,
            "triton.language": language,
            "flaggems_vllm": ModuleType("flaggems_vllm"),
            "flaggems_vllm.runtime": SimpleNamespace(torch_device_fn=None),
            "flaggems_vllm.utils": utils,
            "flaggems_vllm.utils.device_info": SimpleNamespace(
                get_device_capability=lambda: (9, 0), get_sm_count=lambda: 132
            ),
            "attn_pr737_contract": package,
        }
        with patch.dict(sys.modules, shims):
            module = importlib.import_module("attn_pr737_contract.attn_res")
            configs = module.post_configs()
            self.assertEqual(len(configs), 16)
            for bucket in range(3):
                for blocks in range(9):
                    with self.subTest(bucket=bucket, blocks=blocks):
                        pruned = module._prune_post_configs(
                            configs, {}, TOKEN_COUNT_BUCKET=bucket, NUM_BLOCKS=blocks
                        )
                        self.assertTrue(pruned)
                        self.assertFalse(
                            any(c.kwargs["USE_TLE_ASYNC_LOAD"] for c in pruned)
                        )
            for tokens in (1, 64, 256, 4096):
                for blocks in range(9):
                    plan = module._select_fixed_launch_config(tokens, blocks)
                    self.assertEqual(
                        plan.use_persistent_token_loop, tokens >= 256 and blocks >= 3
                    )
                    self.assertEqual(plan.enable_pdl, tokens < 256 and blocks in (2, 3))
            import ast

            tree = ast.parse(
                (ROOT / "vllm_fl/third_party/attn_res_pr737/attn_res.py").read_text()
            )
            update = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_attn_res_fixed_config_kernel"
            )
            self.assertNotIn(
                "libtuner", " ".join(ast.unparse(d) for d in update.decorator_list)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
