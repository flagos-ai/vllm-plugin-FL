# Copyright (c) 2026 BAAI. All rights reserved.

import logging
import unittest
from unittest.mock import patch

import torch


class TestCpuQuantLinearInstaller(unittest.TestCase):
    @staticmethod
    def _bare_layer(cls):
        layer = object.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.weight = torch.nn.Parameter(
            torch.randn(64, 128, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.bias = None
        return layer

    def test_embedding_subclasses_are_not_prepared(self):
        import vllm.model_executor.layers.utils as layer_utils
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm_fl.ops import cpu_quant_linear

        class DerivedEmbedding(VocabParallelEmbedding):
            pass

        prepared = []
        fallback = []

        def prepare(weight):
            prepared.append(weight)
            return lambda x, stored_weight, bias: x

        def original(layer, remove_weight):
            fallback.append(layer)

        with (
            patch.object(layer_utils, "dispatch_cpu_unquantized_gemm", original),
            patch.object(cpu_quant_linear, "_ACTIVE_BACKEND", None),
        ):
            cpu_quant_linear.install_cpu_quantized_linear(
                backend="test-w8",
                prepare_linear=prepare,
                supports_shape=lambda n, k: True,
                include_lm_head=False,
                strict=True,
                logger=logging.getLogger(__name__),
            )
            embedding = self._bare_layer(DerivedEmbedding)
            lm_head = self._bare_layer(ParallelLMHead)
            linear = torch.nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)

            layer_utils.dispatch_cpu_unquantized_gemm(embedding, False)
            layer_utils.dispatch_cpu_unquantized_gemm(lm_head, False)
            layer_utils.dispatch_cpu_unquantized_gemm(linear, False)

        self.assertEqual(fallback, [embedding, lm_head])
        self.assertEqual(prepared, [linear.weight])

    def test_different_backends_cannot_stack_dispatch_patches(self):
        import vllm.model_executor.layers.utils as layer_utils
        from vllm_fl.ops import cpu_quant_linear

        def original(layer, remove_weight):
            return None

        kwargs = {
            "prepare_linear": lambda weight: None,
            "supports_shape": lambda n, k: True,
            "include_lm_head": False,
            "strict": True,
            "logger": logging.getLogger(__name__),
        }
        with (
            patch.object(layer_utils, "dispatch_cpu_unquantized_gemm", original),
            patch.object(cpu_quant_linear, "_ACTIVE_BACKEND", None),
        ):
            cpu_quant_linear.install_cpu_quantized_linear(backend="first", **kwargs)
            with self.assertRaisesRegex(RuntimeError, "already active as first"):
                cpu_quant_linear.install_cpu_quantized_linear(
                    backend="second", **kwargs
                )

    def test_lm_head_is_prepared_only_when_enabled(self):
        import vllm.model_executor.layers.utils as layer_utils
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )
        from vllm_fl.ops import cpu_quant_linear

        for include_lm_head in (False, True):
            with self.subTest(include_lm_head=include_lm_head):
                prepared = []
                fallback = []

                def prepare(weight):
                    prepared.append(weight)
                    return lambda x, stored_weight, bias: x

                def original(layer, remove_weight):
                    fallback.append(layer)

                with (
                    patch.object(
                        layer_utils, "dispatch_cpu_unquantized_gemm", original
                    ),
                    patch.object(cpu_quant_linear, "_ACTIVE_BACKEND", None),
                ):
                    cpu_quant_linear.install_cpu_quantized_linear(
                        backend="test-lm-head",
                        prepare_linear=prepare,
                        supports_shape=lambda n, k: True,
                        include_lm_head=include_lm_head,
                        strict=True,
                        logger=logging.getLogger(__name__),
                    )
                    lm_head = self._bare_layer(ParallelLMHead)
                    layer_utils.dispatch_cpu_unquantized_gemm(lm_head, False)

                self.assertEqual(len(prepared), int(include_lm_head))
                self.assertEqual(len(fallback), int(not include_lm_head))

    def test_failed_backend_initialization_does_not_patch_dispatch(self):
        import vllm.model_executor.layers.utils as layer_utils
        from vllm_fl.ops import cpu_quant_linear

        def original(layer, remove_weight):
            return None

        def fail_initialization():
            raise RuntimeError("native initialization failed")

        with (
            patch.object(layer_utils, "dispatch_cpu_unquantized_gemm", original),
            patch.object(cpu_quant_linear, "_ACTIVE_BACKEND", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "native initialization failed"):
                cpu_quant_linear.install_cpu_quantized_linear(
                    backend="broken",
                    prepare_linear=lambda weight: None,
                    supports_shape=lambda n, k: True,
                    include_lm_head=False,
                    strict=True,
                    logger=logging.getLogger(__name__),
                    initialize=fail_initialization,
                )
            self.assertIsNone(cpu_quant_linear._ACTIVE_BACKEND)
            self.assertIs(layer_utils.dispatch_cpu_unquantized_gemm, original)

if __name__ == "__main__":
    unittest.main()
