# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; no Torch, Triton or GPU imports are required."""

import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "vllm_fl/ops/fused_moe/mxfp4_marlin_reduce.py"
spec = importlib.util.spec_from_file_location("mxfp4_marlin_reduce_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class View:
    def __init__(self, shape, offset=0, storage=None):
        self.shape, self.offset = shape, offset
        self.storage = storage if storage is not None else object()

    def dim(self):
        return len(self.shape)

    def __getitem__(self, rows):
        assert isinstance(rows, slice) and rows.step is None
        return View(
            (rows.stop - rows.start, *self.shape[1:]),
            self.offset + rows.start,
            self.storage,
        )


class Reducer:
    def __init__(self):
        self.calls = []

    def moe_sum(self, input, output):
        assert input.dim() == 3 and output.dim() == 2
        assert output.shape == (input.shape[0], input.shape[2])
        assert input.offset == output.offset
        self.calls.append((input, output))


class ReductionContract(unittest.TestCase):
    def test_boundary_and_large_balanced_partition(self):
        for m in (65535, 65536, 65537, 131070, 131071, 10_000_003):
            with self.subTest(m=m):
                gems, inp, out = Reducer(), View((m, 6, 5120)), View((m, 5120))
                self.assertIsNone(module.safe_moe_sum(gems, inp, out))
                self.assertEqual(sum(i.shape[0] for i, _ in gems.calls), m)
                next_row = 0
                for i, o in gems.calls:
                    self.assertEqual(i.offset, next_row)
                    self.assertIs(i.storage, inp.storage)
                    self.assertIs(o.storage, out.storage)
                    self.assertLessEqual(i.shape[0], 65535)
                    if m > 65535:
                        self.assertGreaterEqual(i.shape[0], 32768)
                    next_row += i.shape[0]
                self.assertEqual(next_row, m)

    def test_exact_65536_two_32768_calls(self):
        gems = Reducer()
        module.safe_moe_sum(gems, View((65536, 6, 5120)), View((65536, 5120)))
        self.assertEqual([i.shape[0] for i, _ in gems.calls], [32768, 32768])

    def test_small_and_empty_pass_original_objects(self):
        for m in (0, 1, 16, 127, 128, 4096, 65535):
            gems, inp, out = Reducer(), View((m, 6, 128)), View((m, 128))
            module.safe_moe_sum(gems, inp, out)
            self.assertEqual(len(gems.calls), 1)
            self.assertIs(gems.calls[0][0], inp)
            self.assertIs(gems.calls[0][1], out)

    def test_bad_global_output_shape_fails_before_any_write(self):
        for shape in ((65537, 5120), (65535, 5120), (65536, 4096)):
            gems = Reducer()
            with self.assertRaisesRegex(AssertionError, "output shape"):
                module.safe_moe_sum(gems, View((65536, 6, 5120)), View(shape))
            self.assertFalse(gems.calls)

    def test_malformed_dimensions_delegate_original_validation(self):
        gems = Reducer()
        with self.assertRaises(AssertionError):
            module.safe_moe_sum(gems, View((65536, 5120)), View((65536, 5120)))


if __name__ == "__main__":
    unittest.main()
