# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Ascend V2 pointer-table override."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _UpstreamBlockTables:
    def __init__(self, pointers, device="cpu"):
        self.device = torch.device(device)
        self.block_table_ptrs = self._make_ptr_tensor(pointers)

    def _make_ptr_tensor(self, pointers):
        return torch.tensor(
            [tensor.data_ptr() for tensor in pointers],
            dtype=torch.uint64,
            device=self.device,
        )


class AscendV2BlockTablesTests(unittest.TestCase):
    def setUp(self):
        modules = {}
        for name in (
            "vllm",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.block_table",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["vllm.v1.worker.gpu.block_table"].BlockTables = _UpstreamBlockTables
        modules_patch = patch.dict(sys.modules, modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        source = (
            Path(__file__).resolve().parents[2]
            / "vllm_fl/worker/ascend_v2_block_table.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ascend_v2_block_table_under_test", source
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_inherited_constructor_uses_npu_supported_int64_storage(self):
        tensors = [torch.zeros(2, 3), torch.zeros(4, 2)]
        block_tables = self.module.AscendBlockTables(tensors)
        self.assertEqual(block_tables.block_table_ptrs.dtype, torch.int64)
        self.assertEqual(
            block_tables.block_table_ptrs.tolist(),
            [tensor.data_ptr() for tensor in tensors],
        )

    def test_pointer_generator_and_empty_table(self):
        table = self.module.AscendBlockTables([])
        pointers = (SimpleNamespace(data_ptr=lambda n=n: n) for n in (0, 128, 4096))
        self.assertEqual(table._make_ptr_tensor(pointers).tolist(), [0, 128, 4096])
        self.assertEqual(table.block_table_ptrs.shape, (0,))
        self.assertEqual(table.block_table_ptrs.dtype, torch.int64)

    def test_high_bit_addresses_preserve_all_pointer_bits(self):
        addresses = [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
        pointers = [SimpleNamespace(data_ptr=lambda n=n: n) for n in addresses]
        table = self.module.AscendBlockTables(pointers)
        signed = table.block_table_ptrs.tolist()
        self.assertEqual(signed, [0, (1 << 63) - 1, -(1 << 63), -1])
        self.assertEqual([value & ((1 << 64) - 1) for value in signed], addresses)

    def test_invalid_pointer_range_is_rejected_before_tensor_creation(self):
        for address in (-1, 1 << 64):
            with (
                self.subTest(address=address),
                self.assertRaisesRegex(ValueError, "64-bit pointer range"),
            ):
                self.module._signed_pointer(address)

    def test_generic_upstream_class_is_unchanged(self):
        tensor = torch.zeros(2)
        generic = _UpstreamBlockTables([tensor])
        ascend = self.module.AscendBlockTables([tensor])
        self.assertEqual(generic.block_table_ptrs.dtype, torch.uint64)
        self.assertEqual(ascend.block_table_ptrs.dtype, torch.int64)
        self.assertIs(self.module.AscendBlockTables.__bases__[0], _UpstreamBlockTables)


if __name__ == "__main__":
    unittest.main()
