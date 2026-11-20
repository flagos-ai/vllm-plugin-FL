# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU tests for the single scoped sampler hook."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class SamplingHookTests(unittest.TestCase):
    def run_case(self, enabled=True, bad=False):
        modules = {}
        for path in (
            "vllm",
            "vllm.v1",
            "vllm.v1.sample",
            "vllm.v1.sample.ops",
            "vllm.v1.sample.ops.topk_topp_sampler",
            "vllm_fl",
            "vllm_fl.minimax_m3_config",
            "vllm_fl.ops",
            "vllm_fl.ops.minimax_m3",
            "vllm_fl.ops.minimax_m3.sampling",
        ):
            obj = types.ModuleType(path)
            obj.__path__ = []
            modules[path] = obj
            parent, _, name = path.rpartition(".")
            if parent:
                setattr(modules[parent], name, obj)
        modules["vllm_fl.minimax_m3_config"].is_enabled = lambda: enabled

        def original(logits, k, p):
            return logits

        def invalid(logits):
            return logits

        original.__module__ = (
            "vllm_fl.dispatch.backends.vendor.metax.patches.topk_topp_sampler"
        )
        current = invalid if bad else original
        target = modules["vllm.v1.sample.ops.topk_topp_sampler"]
        target.apply_top_k_top_p = current
        replacement = lambda logits, k, p: logits
        modules["vllm_fl.ops.minimax_m3.sampling"].apply_top_k_top_p = replacement
        path = ROOT / "vllm_fl/patches/minimax_m3_metax/sampling.py"
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "_m3_sampling_hook_test", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if bad and enabled:
                with self.assertRaisesRegex(RuntimeError, "pinned MetaX"):
                    module.install()
                self.assertIs(target.apply_top_k_top_p, current)
            else:
                module.install()
                expected = replacement if enabled else current
                self.assertIs(target.apply_top_k_top_p, expected)
                module.install()
                self.assertIs(target.apply_top_k_top_p, expected)

    def test_other_models_are_unchanged(self):
        self.run_case(enabled=False)

    def test_idempotent_install(self):
        self.run_case()

    def test_signature_failure_preserves_original(self):
        self.run_case(bad=True)


if __name__ == "__main__":
    unittest.main()
