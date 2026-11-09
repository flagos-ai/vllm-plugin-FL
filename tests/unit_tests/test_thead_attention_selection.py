"""CPU checks for PPU attention kind selection, independent of GPU imports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class TheadAttentionSelectionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        base_spec = importlib.util.spec_from_file_location(
            "thead_test_base", root / "vllm_fl/dispatch/backends/base.py"
        )
        base = importlib.util.module_from_spec(base_spec)
        base_spec.loader.exec_module(base)
        torch = ModuleType("torch")
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": torch,
                "vllm_fl.dispatch.backends.base": base,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            "thead_test_backend",
            root / "vllm_fl/dispatch/backends/vendor/thead/thead.py",
        )
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        self.backend = backend.TheadBackend()

    def test_dense_mla_uses_an_mla_backend(self):
        self.assertEqual(
            self.backend.attention_backend(use_mla=True),
            "vllm_fl.dispatch.backends.vendor.thead.impl.mla.TheadMLABackend",
        )

    def test_standard_attention_retains_ppu_fa3(self):
        self.assertEqual(
            self.backend.attention_backend(),
            "vllm_fl.dispatch.backends.vendor.thead.impl.attention.TheadFlashAttentionBackend",
        )

    def test_sparse_requires_mla(self):
        with self.assertRaises(ValueError):
            self.backend.attention_backend(use_sparse=True)

    def test_sparse_is_not_silently_routed_to_dense_attention(self):
        with self.assertRaises(NotImplementedError):
            self.backend.attention_backend(use_mla=True, use_sparse=True)


if __name__ == "__main__":
    unittest.main()
