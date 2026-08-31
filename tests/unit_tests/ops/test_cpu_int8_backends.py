# Copyright (c) 2026 BAAI. All rights reserved.

import unittest

import torch


class TestCpuInt8Backends(unittest.TestCase):
    def test_torchpack_matches_dequantized_reference(self):
        from vllm_fl.ops import cpu_int8_pack as op

        torch.manual_seed(20260804)
        n, k = 64, 128
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        quantized, scale = op._quantize_int8(weight)
        linear = op._make_cpu_linear(quantized, scale, n, k)
        dequantized = quantized.float() * scale.float()[:, None]
        for m in (1, 3, 7):
            x = torch.randn(m, k, dtype=torch.bfloat16)
            actual = linear(x, weight, None).float()
            expected = x.float() @ dequantized.T
            relative_error = torch.linalg.vector_norm(actual - expected) / (
                torch.linalg.vector_norm(expected) + 1e-12
            )
            self.assertLess(float(relative_error), 0.02)


if __name__ == "__main__":
    unittest.main()
