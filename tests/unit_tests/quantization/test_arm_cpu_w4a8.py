# Copyright (c) 2026 BAAI. All rights reserved.

import platform
import unittest
from unittest.mock import patch

import flag_gems
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs

from vllm_fl.quantization.arm_cpu_w4a8 import install_arm_cpu_packed_w4a8

_IS_ARM_CPU_RUNTIME = (
    platform.machine().lower() in {"aarch64", "arm64"}
    and flag_gems.vendor_name == "arm"
)


@unittest.skipUnless(
    _IS_ARM_CPU_RUNTIME,
    "requires an ARM64 host with the FlagGems ARM CPU backend",
)
class TestArmCpuPackedW4A8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
            Dynamic4bitLinearKernel,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors import (
            compressed_tensors as config_module,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w4a8_int as scheme_module,
        )

        cls.scheme_cls = scheme_module.CompressedTensorsW4A8Int
        cls.config_cls = config_module.CompressedTensorsConfig
        cls.kernel_cls = Dynamic4bitLinearKernel
        cls.originals = {
            "scheme_init": cls.scheme_cls.__init__,
            "scheme_create_weights": cls.scheme_cls.create_weights,
            "config_get_scheme": cls.config_cls._get_scheme_from_parts,
            "kernel_process": cls.kernel_cls.process_weights_after_loading,
            "kernel_apply": cls.kernel_cls.apply_weights,
        }
        cls.first_install = install_arm_cpu_packed_w4a8()

    @classmethod
    def tearDownClass(cls):
        if not cls.first_install:
            return

        cls.scheme_cls.__init__ = cls.originals["scheme_init"]
        cls.scheme_cls.create_weights = cls.originals["scheme_create_weights"]
        cls.config_cls._get_scheme_from_parts = cls.originals["config_get_scheme"]
        cls.kernel_cls.process_weights_after_loading = cls.originals["kernel_process"]
        cls.kernel_cls.apply_weights = cls.originals["kernel_apply"]
        for owner, attribute in (
            (cls.scheme_cls, "_vllm_fl_arm_packed_w4a8"),
            (cls.scheme_cls, "_vllm_fl_arm_original_init"),
            (cls.scheme_cls, "_vllm_fl_arm_original_create_weights"),
            (cls.config_cls, "_vllm_fl_arm_original_get_scheme"),
            (cls.kernel_cls, "_vllm_fl_arm_w4a8"),
            (cls.kernel_cls, "_vllm_fl_arm_original_process"),
            (cls.kernel_cls, "_vllm_fl_arm_original_apply"),
        ):
            if attribute in owner.__dict__:
                delattr(owner, attribute)

    def test_installer_is_idempotent(self):
        self.assertFalse(install_arm_cpu_packed_w4a8())

    def test_packed_checkpoint_scheme_preserves_metadata(self):
        from vllm.model_executor.layers.quantization.compressed_tensors import (
            compressed_tensors as config_module,
        )

        config = object.__new__(config_module.CompressedTensorsConfig)
        config.quant_format = CompressionFormat.pack_quantized.value
        weight_quant = QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            group_size=128,
            strategy="group",
            dynamic=False,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="token",
            dynamic=True,
        )

        scheme = config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format=CompressionFormat.pack_quantized.value,
            layer_name="model.layers.0.linear_attn.in_proj_qkvz",
        )
        layer = torch.nn.Module()
        with (
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
        ):
            scheme.create_weights(
                layer,
                output_size=48,
                input_size=5120,
                output_partition_sizes=[48],
                input_size_per_partition=5120,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )

        self.assertEqual(layer.weight_packed.shape, (48, 640))
        self.assertEqual(layer.weight_packed.dtype, torch.int32)
        self.assertEqual(layer.weight_scale.shape, (48, 40))
        self.assertEqual(layer.weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer.weight_shape.shape, (2,))
        self.assertEqual(layer.weight_shape.dtype, torch.int64)
        self.assertEqual(scheme.kernel.config.group_size, 128)
        self.assertFalse(scheme.kernel.config.zero_points)
        self.assertTrue(scheme.kernel._vllm_fl_arm_packed_checkpoint)

        layer.weight_packed.data.zero_()
        layer.weight_scale.data.fill_(0.01)
        layer.weight_shape.data.copy_(torch.tensor([48, 5120]))
        scheme.kernel.process_weights_after_loading(layer)

        self.assertEqual(layer.weight_packed.dtype, torch.uint8)
        self.assertEqual(scheme.kernel._vllm_fl_arm_w4a8_shape, (48, 5120))
        output = scheme.kernel.apply_weights(
            layer,
            torch.zeros((1, 5120), dtype=torch.bfloat16),
        )
        self.assertEqual(output.shape, (1, 48))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_unpacked_checkpoint_keeps_stock_kernel_path(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w4a8_int as scheme_module,
        )

        scheme = scheme_module.CompressedTensorsW4A8Int(
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        layer = torch.nn.Module()
        layer.bias = None
        with (
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
        ):
            scheme.create_weights(
                layer,
                output_size=48,
                input_size=5120,
                output_partition_sizes=[48],
                input_size_per_partition=5120,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )

        self.assertNotIn(
            "_vllm_fl_arm_packed_checkpoint",
            vars(scheme.kernel),
        )
        layer.weight_packed.data.zero_()
        layer.weight_scale.data.fill_(0.01)
        scheme.kernel.process_weights_after_loading(layer)
        self.assertNotIn("_vllm_fl_arm_w4a8_shape", vars(scheme.kernel))

    def test_packed_checkpoint_rejects_invalid_group_partition(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w4a8_int as scheme_module,
        )

        scheme = scheme_module.CompressedTensorsW4A8Int(
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        with self.assertRaisesRegex(ValueError, "not divisible"):
            scheme.create_weights(
                torch.nn.Module(),
                output_size=48,
                input_size=130,
                output_partition_sizes=[48],
                input_size_per_partition=130,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )


if __name__ == "__main__":
    unittest.main()
